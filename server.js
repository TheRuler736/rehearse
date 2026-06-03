import express from "express";
import dotenv from "dotenv";
import crypto from "crypto";

dotenv.config();

const {
  SENDBLUE_API_KEY,
  SENDBLUE_API_SECRET,
  SENDBLUE_FROM_NUMBER, // optional: your Sendblue line to send from (e.g. +1...)
  GROQ_API_KEY,
  GROQ_MODEL = "llama-3.1-8b-instant",
  DEFAULT_TZ = "America/Los_Angeles", // used until a user tells the coach their timezone
  ADMIN_KEY, // guards /admin/* (falls back to SENDBLUE_API_SECRET)
  STRIPE_SECRET_KEY, // lets the server ask Stripe who's actually subscribed
  STRIPE_WEBHOOK_SECRET, // set once Stripe is wired up; verifies webhook signatures
  SUBSCRIBE_URL = "https://rehearse-143f.onrender.com", // where unpaid users go to subscribe
  PRICE = "$29.99/month",
  LEAD_TTL_HOURS = "24", // delete an unpaid lead's Sendblue contact after this much inactivity
  PORT = 3000,
} = process.env;

if (!SENDBLUE_API_KEY || !SENDBLUE_API_SECRET || !GROQ_API_KEY) {
  console.error("Missing env vars. Need SENDBLUE_API_KEY, SENDBLUE_API_SECRET, and GROQ_API_KEY.");
  process.exit(1);
}

const MAX_TURNS = 40; // cap stored history (user+assistant messages) per number

const app = express();
// Stripe webhooks need the raw body for signature verification, so parse that
// route as raw and everything else as JSON.
app.use("/stripe/webhook", express.raw({ type: "*/*" }));
app.use((req, res, next) => (req.originalUrl === "/stripe/webhook" ? next() : express.json()(req, res, next)));
app.use(express.static(".")); // serves index.html (the landing page)

/**
 * The coach's personality + rules. Tuned for texting: short, one question at a
 * time, brief feedback, encouraging. Sent to the model as the system message.
 */
const BASE_PROMPT = `You are Rehearse, a professional and supportive AI interview coach speaking with a subscriber over text message (iMessage/SMS).

STYLE:
- Professional, clear, and encouraging — polished but warm. No slang, no fluff.
- Keep every reply SHORT — this is texting. Usually 1-4 sentences.
- Plain text only. No markdown, bullet symbols, or headers.

HOW A SESSION WORKS:
- When the user names a company (e.g. "Apple") or a role (e.g. "product manager"), begin a tailored mock interview.
- Open with one brief line confirming what you'll practice, then ask the FIRST question.
- Ask ONE question at a time and wait for their answer.
- After each answer, give brief, specific, professional feedback (what was strong + one concrete improvement), then ask the next question.
- Mix behavioral and role-relevant questions appropriate to the company/role.
- If they say "stop", "new", "restart", or name a different company, switch gracefully.
- If the first message is vague (e.g. "hi"), greet them professionally and ask which company or role they'd like to practice for.

Maintain a focused, momentum-building session that measurably improves their answers.`;

/**
 * Prompt for people who have NOT subscribed. The coach stays professional and
 * helpful, explains the value, and guides them to subscribe — but withholds the
 * actual coaching (no mock questions, no feedback) until they pay.
 */
function buildSalesPrompt(number) {
  // Personalize the subscribe link with the texter's number so Stripe ties the
  // payment back to the exact phone they message from (matched via client_reference_id).
  const link = number
    ? `${SUBSCRIBE_URL}${SUBSCRIBE_URL.includes("?") ? "&" : "?"}client_reference_id=${String(number).replace(/\D/g, "")}`
    : SUBSCRIBE_URL;
  return `You are Rehearse, a professional AI interview coach, texting someone over iMessage who has NOT yet subscribed.

YOUR GOAL: Be genuinely helpful and professional, build their interest, and gently guide them to subscribe — without giving away the coaching itself.

STRICT RULES (do not break these):
- Do NOT run a mock interview, ask practice interview questions, evaluate answers, or give interview tips/feedback. That is for subscribers only.
- If they ask you to practice or coach them, warmly acknowledge it and explain that full coaching is included with a Rehearse subscription, then invite them to start.
- Keep replies short (1-3 sentences), polished, warm, and professional. Plain text only.

WHAT TO CONVEY over the conversation (naturally, not all at once):
- Rehearse is a professional AI interview coach that lives in your texts: realistic mock interviews for any company or role, instant specific feedback, proactive reminders before your real interview, and daily practice questions.
- It is ${PRICE} for unlimited everything, cancel anytime.
- To begin, they subscribe here: ${link}

STYLE: Confident, encouraging, never pushy or salesy. Sell subtly — be so helpful that the value is obvious. When they seem interested, make subscribing the clear next step and share the link.`;
}

/**
 * Build the full system prompt for a given user, injecting the current time (in
 * their timezone) and the rules for emitting scheduling directives. The model
 * writes a normal reply AND, when the user wants something scheduled, appends a
 * hidden [[...]] directive that the server parses out and acts on.
 */
function buildSystemPrompt(number) {
  const tz = tzFor(number);
  const now = new Date().toLocaleString("en-US", {
    timeZone: tz, weekday: "long", year: "numeric", month: "long",
    day: "numeric", hour: "numeric", minute: "2-digit",
  });
  return `${BASE_PROMPT}

CURRENT TIME: ${now} (timezone ${tz}).

SCHEDULING:
You can set reminders and recurring daily practice for the user. If they ask to be reminded about an interview, want a daily practice question, mention a real interview date/time, or ask to cancel, acknowledge it naturally and state the EXACT day and time you'll use (e.g. "Got it — I'll text you the night before and an hour before, this Friday."). Keep it short. The system schedules it automatically; do not output any special codes.`;
}

/**
 * A separate, strict prompt used only to EXTRACT a scheduling action from the
 * conversation. A small model follows one narrow job far more reliably than
 * coaching + emitting machine syntax in the same reply.
 */
function buildExtractionPrompt(number) {
  const tz = tzFor(number);
  const now = new Date().toLocaleString("en-US", {
    timeZone: tz, weekday: "long", year: "numeric", month: "long",
    day: "numeric", hour: "numeric", minute: "2-digit",
  });
  return `You extract scheduling actions from an interview-coaching text chat.
CURRENT TIME: ${now} (timezone ${tz}).
Look at the conversation, focusing on the user's most recent request, and output EXACTLY ONE line — nothing else:
- Reminder about a specific interview date/time:
  [[REMIND interview | <ISO8601 with timezone offset> | <short note>]]
- A one-off reminder at a specific time:
  [[REMIND once | <ISO8601 with offset> | <the message to text them>]]
- A recurring daily practice question:
  [[REMIND daily | <HH:MM 24-hour> | <role or company>]]
- Cancel their reminders: [[REMIND cancel]]
- They stated their timezone: [[TZ <IANA timezone name>]]
- No scheduling request at all: NONE
Resolve relative dates/times ("Friday", "tomorrow at 9", "in 2 hours", "tonight") against CURRENT TIME, using the correct UTC offset. Output ONLY the single directive line or the word NONE.

Examples:
"remind me my Apple interview is friday at 2pm" -> [[REMIND interview | 2026-06-06T14:00:00-07:00 | Apple interview]]
"send me a question every morning at 8" -> [[REMIND daily | 08:00 | general interview]]
"actually stop the reminders" -> [[REMIND cancel]]
"how did I do on that answer?" -> NONE`;
}

/**
 * In-memory conversation store, keyed by phone number.
 * NOTE: resets if the server restarts. Fine for v1 on a single machine.
 * For production, swap this Map for Redis or a database.
 */
const conversations = new Map();

function getHistory(number) {
  if (!conversations.has(number)) conversations.set(number, []);
  return conversations.get(number);
}

/**
 * Scheduling state (in-memory — resets on restart/redeploy; a heartbeat keeps the
 * server awake so it persists between deploys). A tiny DB would make it bulletproof.
 */
const timezones = new Map(); // number -> IANA timezone
const reminders = []; // { id, number, line, kind:"once"|"daily", fireAt?, recurAt?, lastSent?, context?, note? }
let reminderSeq = 1;

function tzFor(number) {
  return timezones.get(number) || DEFAULT_TZ;
}
function isValidTz(tz) {
  try { new Intl.DateTimeFormat("en-US", { timeZone: tz }); return true; } catch { return false; }
}
/** Current wall-clock { date:"YYYY-MM-DD", hhmm:"HH:MM" } in a timezone. */
function nowInTz(tz) {
  const p = new Intl.DateTimeFormat("en-CA", {
    timeZone: tz, year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", hour12: false,
  }).formatToParts(new Date()).reduce((a, x) => ((a[x.type] = x.value), a), {});
  const hour = p.hour === "24" ? "00" : p.hour;
  return { date: `${p.year}-${p.month}-${p.day}`, hhmm: `${hour}:${p.minute}` };
}

/**
 * Subscription state. A number is "paid" once Stripe confirms an active
 * subscription (or an admin grants it). In-memory for now — Stripe is the source
 * of truth, so a persistent store can be added when we wire Stripe fully.
 */
const paidNumbers = new Set();
function isPaid(number) { return paidNumbers.has(number); }
function setPaid(number, paid) { paid ? paidNumbers.add(number) : paidNumbers.delete(number); }

/** Normalize a phone string to E.164-ish for consistent matching. */
function normalizePhone(p) {
  if (!p) return p;
  const d = String(p).replace(/[^\d]/g, "");
  if (d.length === 10) return `+1${d}`;
  return `+${d}`;
}

/** Minimal Stripe REST helper (no SDK). */
async function stripeApi(path, { method = "GET", body } = {}) {
  if (!STRIPE_SECRET_KEY) throw new Error("STRIPE_SECRET_KEY not set");
  const res = await fetch("https://api.stripe.com/v1/" + path, {
    method,
    headers: {
      Authorization: "Bearer " + STRIPE_SECRET_KEY,
      ...(body ? { "Content-Type": "application/x-www-form-urlencoded" } : {}),
    },
    body,
  });
  const j = await res.json();
  if (j.error) throw new Error(j.error.message);
  return j;
}

/**
 * Rebuild the paid list from Stripe (the source of truth). Runs on startup and
 * periodically, so paid customers stay unlocked even after the server restarts.
 * Relies on each customer's metadata.phone, which the webhook stamps at checkout.
 */
async function syncPaidFromStripe() {
  if (!STRIPE_SECRET_KEY) return;
  try {
    const params = new URLSearchParams({ status: "active", limit: "100" });
    params.append("expand[]", "data.customer");
    const subs = await stripeApi("subscriptions?" + params.toString());
    let n = 0;
    for (const sub of subs.data || []) {
      const c = sub.customer || {};
      const phone = normalizePhone(c.metadata?.phone || c.phone || "");
      if (phone && phone !== "+") { setPaid(phone, true); n++; }
    }
    console.log(`Stripe sync: ${n} active subscriber(s) loaded`);
  } catch (err) {
    console.warn("Stripe sync failed:", err.message);
  }
}

/** Verify a Stripe webhook signature (so we don't need the Stripe SDK). */
function verifyStripe(rawBody, sigHeader, secret) {
  const parts = Object.fromEntries(String(sigHeader).split(",").map((kv) => kv.split("=")));
  const signed = `${parts.t}.${rawBody.toString("utf8")}`;
  const expected = crypto.createHmac("sha256", secret).update(signed).digest("hex");
  const a = Buffer.from(expected);
  const b = Buffer.from(parts.v1 || "");
  if (a.length !== b.length || !crypto.timingSafeEqual(a, b)) throw new Error("signature mismatch");
  if (Math.abs(Date.now() / 1000 - Number(parts.t)) > 300) throw new Error("timestamp out of tolerance");
  return JSON.parse(rawBody.toString("utf8"));
}

/** Call Groq (OpenAI-compatible) with the running history; return the reply text. */
async function askModel(history, systemPrompt, { temperature = 0.8, maxTokens = 300 } = {}) {
  const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${GROQ_API_KEY}`,
    },
    body: JSON.stringify({
      model: GROQ_MODEL,
      messages: [{ role: "system", content: systemPrompt }, ...history],
      temperature,
      max_tokens: maxTokens,
    }),
  });

  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Groq ${res.status}: ${detail}`);
  }

  const data = await res.json();
  const text = data?.choices?.[0]?.message?.content?.trim();
  return text || "Sorry, I blanked for a second — can you say that again?";
}

/** Send a text back to the user via Sendblue. fromNumber = which of your lines to send from. */
async function sendText(number, content, fromNumber) {
  const payload = { number, content };
  // Sendblue requires telling it which line to send from on multi-line/free plans.
  const from = fromNumber || SENDBLUE_FROM_NUMBER;
  if (from) payload.from_number = from;

  const res = await fetch("https://api.sendblue.co/api/send-message", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "sb-api-key-id": SENDBLUE_API_KEY,
      "sb-api-secret-key": SENDBLUE_API_SECRET,
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Sendblue ${res.status}: ${detail}`);
  }
  return res.json();
}

/**
 * Best-effort Sendblue notification (typing indicator / read receipt).
 * These are cosmetic, iMessage-only, and must never break the actual reply,
 * so failures are logged and swallowed rather than thrown.
 * NOTE: read receipts (mark-read) require Sendblue to enable them on the account.
 */
async function sbNotify(path, number, fromNumber) {
  try {
    const body = { number };
    const from = fromNumber || SENDBLUE_FROM_NUMBER;
    if (from) body.from_number = from;
    const res = await fetch(`https://api.sendblue.co/api/${path}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "sb-api-key-id": SENDBLUE_API_KEY,
        "sb-api-secret-key": SENDBLUE_API_SECRET,
      },
      body: JSON.stringify(body),
    });
    if (!res.ok) console.warn(`${path} ${res.status}: ${(await res.text()).slice(0, 200)}`);
  } catch (err) {
    console.warn(`${path} error:`, err.message);
  }
}

// ---------------------------------------------------------------------------
// Contact lifecycle: verify subscribers (so we can text them) and prune unpaid
// leads (free-plan contact slots are limited).
// ---------------------------------------------------------------------------

const lastSeen = new Map(); // number -> last inbound timestamp (ms)
const LEAD_TTL_MS = (Number(LEAD_TTL_HOURS) || 24) * 3600e3;

/** Verify a Sendblue contact so proactive messages can reach them. Best-effort. */
async function verifyContact(number) {
  try {
    const res = await fetch("https://api.sendblue.co/api/v2/contacts/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json", "sb-api-key-id": SENDBLUE_API_KEY, "sb-api-secret-key": SENDBLUE_API_SECRET },
      body: JSON.stringify({ number }),
    });
    if (!res.ok) console.warn(`verify ${res.status}: ${(await res.text()).slice(0, 160)}`);
    else console.log("✔ verified contact:", number);
  } catch (err) {
    console.warn("verify error:", err.message);
  }
}

/** Delete a Sendblue contact (frees a slot on the free plan). Best-effort. */
async function deleteContact(number) {
  try {
    const res = await fetch(`https://api.sendblue.co/api/v2/contacts/${encodeURIComponent(number)}`, {
      method: "DELETE",
      headers: { "sb-api-key-id": SENDBLUE_API_KEY, "sb-api-secret-key": SENDBLUE_API_SECRET },
    });
    if (!res.ok && res.status !== 404) console.warn(`delete contact ${res.status}: ${(await res.text()).slice(0, 160)}`);
  } catch (err) {
    console.warn("delete contact error:", err.message);
  }
}

/** Remove unpaid leads that have gone quiet, so their contact slot is freed. */
async function cleanupStaleLeads() {
  const now = Date.now();
  for (const [number, ts] of lastSeen) {
    if (isPaid(number)) continue;         // never remove paying customers
    if (now - ts < LEAD_TTL_MS) continue; // still within the conversion window
    await deleteContact(number);
    conversations.delete(number);
    lastSeen.delete(number);
    console.log("🧹 removed stale unpaid lead:", number);
  }
}

// ---------------------------------------------------------------------------
// Scheduling: parse the model's hidden [[...]] directives, store reminders, and
// fire them on each /cron/tick (driven by an external heartbeat).
// ---------------------------------------------------------------------------

const DIRECTIVE_RE = /\[\[\s*(?:REMIND|TZ)\b[^\]]*\]\]/gi;

function addReminder(r) {
  if (r.kind === "once" && (!r.fireAt || r.fireAt <= Date.now())) return; // skip past one-offs
  reminders.push({ id: reminderSeq++, ...r });
}

/**
 * Read any [[...]] directives the model emitted, create/cancel reminders or set
 * the timezone, and return the reply with those directives stripped out.
 */
function applyDirectives(number, line, text) {
  for (const raw of text.match(DIRECTIVE_RE) || []) {
    const inner = raw.replace(/^\[\[\s*|\s*\]\]$/g, "").trim();

    if (/^TZ\b/i.test(inner)) {
      const tz = inner.replace(/^TZ\s+/i, "").trim();
      if (isValidTz(tz)) { timezones.set(number, tz); console.log(`tz[${number}] = ${tz}`); }
      continue;
    }

    const parts = inner.replace(/^REMIND\s+/i, "").split("|").map((s) => s.trim());
    const kind = (parts[0] || "").toLowerCase();

    if (kind === "cancel") {
      for (let i = reminders.length - 1; i >= 0; i--) if (reminders[i].number === number) reminders.splice(i, 1);
      console.log(`reminders cleared for ${number}`);
    } else if (kind === "interview") {
      const when = Date.parse(parts[1]);
      const note = parts[2] || "your interview";
      if (!isNaN(when)) {
        addReminder({ number, line, kind: "once", fireAt: when - 24 * 3600e3,
          note: `Heads up — ${note} is tomorrow! Want to run a few practice questions tonight? 💪` });
        addReminder({ number, line, kind: "once", fireAt: when - 3600e3,
          note: `${note} is in about an hour! Want a quick 2-minute warm-up? You've got this 🙌` });
      }
    } else if (kind === "once") {
      const when = Date.parse(parts[1]);
      if (!isNaN(when)) addReminder({ number, line, kind: "once", fireAt: when, note: parts[2] || "Reminder 👋" });
    } else if (kind === "daily") {
      const at = /^\d{1,2}:\d{2}$/.test(parts[1] || "") ? parts[1] : "09:00";
      const [h, m] = at.split(":");
      addReminder({ number, line, kind: "daily", recurAt: `${String(+h).padStart(2, "0")}:${m}`,
        context: parts[2] || "general interview", lastSent: null });
    }
  }
  return text.replace(DIRECTIVE_RE, "").replace(/\n{3,}/g, "\n\n").trim();
}

/** Remove any stray directives from a reply without acting on them. */
function stripDirectives(text) {
  return text.replace(DIRECTIVE_RE, "").replace(/\n{3,}/g, "\n\n").trim();
}

/**
 * Second pass: detect and apply a scheduling action from the conversation.
 * Cheap keyword gate first so we don't make an extra model call on every text.
 */
async function extractAndApply(number, line, history) {
  const lastUser = [...history].reverse().find((m) => m.role === "user")?.content || "";
  const schedulingLikely =
    /\b(remind|reminder|schedule|daily|every\s+(morning|day|night|week)|each\s+(morning|day)|tomorrow|tonight|tmrw|next\s+(week|mon|tue|wed|thu|fri|sat|sun)|interview\s+is|\d\s*(am|pm)|\d{1,2}:\d{2}|noon|midnight|timezone|time\s*zone|cancel|stop\s+remind)\b/i.test(lastUser);
  if (!schedulingLikely) return;
  try {
    const convo = history
      .slice(-6)
      .map((m) => `${m.role === "user" ? "User" : "Coach"}: ${m.content}`)
      .join("\n");
    const directive = await askModel(
      [{ role: "user", content: `Conversation:\n${convo}\n\nExtract the scheduling action (or NONE):` }],
      buildExtractionPrompt(number),
      { temperature: 0, maxTokens: 120 }
    );
    console.log(`extract[${number}]: ${directive}`);
    if (directive && directive.match(DIRECTIVE_RE)) applyDirectives(number, line, directive);
  } catch (err) {
    console.warn("extract error:", err.message);
  }
}

/** Send a proactive (coach-initiated) text and record it in the conversation. */
async function deliverProactive(r, message) {
  await sbNotify("send-typing-indicator", r.number, r.line);
  await sendText(r.number, message, r.line);
  const h = getHistory(r.number);
  h.push({ role: "assistant", content: message });
  if (h.length > MAX_TURNS) h.splice(0, h.length - MAX_TURNS);
  console.log(`⏰ → ${r.number}: ${message}`);
}

/** Generate one fresh practice question for a daily reminder. */
async function generateDailyQuestion(context) {
  try {
    const q = await askModel(
      [{ role: "user", content: `Give me ONE concise ${context} interview question to practice today. Output only the question.` }],
      "You are an interview coach. Reply with a single short interview question in plain text, no preamble."
    );
    return `Morning! ☀️ Daily practice — ${q}`;
  } catch {
    return `Morning! ☀️ Daily practice: tell me about a recent challenge you handled well and what you learned.`;
  }
}

/** Fire every reminder that is due. Called by /cron/tick. */
async function runDueReminders() {
  const now = Date.now();
  let sent = 0;
  for (let i = reminders.length - 1; i >= 0; i--) {
    const r = reminders[i];
    try {
      if (r.kind === "once" && r.fireAt <= now) {
        await deliverProactive(r, r.note);
        reminders.splice(i, 1);
        sent++;
      } else if (r.kind === "daily") {
        const { date, hhmm } = nowInTz(tzFor(r.number));
        if (hhmm >= r.recurAt && r.lastSent !== date) {
          // Only mark "sent today" if the send actually succeeds, so a failure retries next tick.
          await deliverProactive(r, await generateDailyQuestion(r.context));
          r.lastSent = date;
          sent++;
        }
      }
    } catch (err) {
      console.error("reminder fire error:", err.message);
    }
  }
  return { sent, pending: reminders.length };
}

/**
 * Sendblue posts every inbound message here.
 * Set this URL as your "Receive URL" in the Sendblue dashboard.
 */
app.post("/webhook/sendblue", async (req, res) => {
  // Acknowledge immediately so Sendblue doesn't retry while we think.
  res.sendStatus(200);

  try {
    console.log("inbound payload:", JSON.stringify(req.body));
    const { content, from_number, is_outbound } = req.body || {};

    // Ignore our own outbound messages and empty payloads.
    if (is_outbound || !content || !from_number) return;

    console.log(`← ${from_number}: ${content}`);
    lastSeen.set(from_number, Date.now()); // track activity for lead cleanup

    // Which of our lines received this? Reply from that same line.
    const line = req.body.to_number || req.body.number || SENDBLUE_FROM_NUMBER;

    // Mark their message "Read" and show the typing "…" bubble while we think.
    await sbNotify("mark-read", from_number, line);
    await sbNotify("send-typing-indicator", from_number, line);

    // Paywall: subscribers get full coaching; everyone else gets the sales flow.
    const paid = isPaid(from_number);
    const history = getHistory(from_number);
    history.push({ role: "user", content });

    const systemPrompt = paid ? buildSystemPrompt(from_number) : buildSalesPrompt(from_number);
    const reply = stripDirectives(await askModel(history, systemPrompt));
    history.push({ role: "assistant", content: reply });

    // Trim old turns to cap memory use.
    if (history.length > MAX_TURNS) {
      history.splice(0, history.length - MAX_TURNS);
    }

    // The model answers in ~1s, so the typing bubble would flash by unseen.
    // Hold briefly (scaled to reply length, capped) so the "…" is actually visible
    // and the coach feels like it's typing rather than firing back instantly.
    const typingMs = Math.min(1200 + reply.length * 25, 4000);
    await new Promise((r) => setTimeout(r, typingMs));

    await sendText(from_number, reply, line);
    console.log(`→ ${from_number} (${paid ? "paid" : "free"}): ${reply}`);

    // Scheduling is a paid feature — only run the extractor for subscribers.
    if (paid) await extractAndApply(from_number, line, history);
  } catch (err) {
    console.error("webhook error:", err.message);
  }
});

// Local test endpoint: POST {"number":"+1...","content":"Apple"} to simulate an
// inbound text without going through Sendblue. Replies in the JSON response.
app.post("/test", async (req, res) => {
  try {
    const { number = "+10000000000", content, line, paid: paidFlag } = req.body || {};
    if (!content) return res.status(400).json({ error: "content required" });
    const paid = paidFlag !== undefined ? !!paidFlag : isPaid(number);
    const history = getHistory(number);
    history.push({ role: "user", content });
    const reply = stripDirectives(await askModel(history, paid ? buildSystemPrompt(number) : buildSalesPrompt(number)));
    history.push({ role: "assistant", content: reply });
    if (paid) await extractAndApply(number, line, history);
    res.json({ reply, paid, reminders: reminders.filter((r) => r.number === number), tz: tzFor(number) });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ---------------------------------------------------------------------------
// Subscription endpoints
// ---------------------------------------------------------------------------

const ADMIN_SECRET = ADMIN_KEY || SENDBLUE_API_SECRET;
const adminOK = (req) => (req.query.key || req.headers["x-admin-key"]) === ADMIN_SECRET;

// Manually grant/revoke access (for testing now; Stripe handles it in production).
app.post("/admin/grant", (req, res) => {
  if (!adminOK(req)) return res.sendStatus(403);
  const n = normalizePhone(req.query.number || req.body?.number);
  if (!n) return res.status(400).json({ error: "number required" });
  setPaid(n, true);
  res.json({ ok: true, number: n, paid: true });
});
app.post("/admin/revoke", (req, res) => {
  if (!adminOK(req)) return res.sendStatus(403);
  const n = normalizePhone(req.query.number || req.body?.number);
  if (!n) return res.status(400).json({ error: "number required" });
  setPaid(n, false);
  res.json({ ok: true, number: n, paid: false });
});
app.get("/admin/status", (req, res) => {
  if (!adminOK(req)) return res.sendStatus(403);
  res.json({ paid: [...paidNumbers] });
});
// Force a re-sync from Stripe (handy for testing the source-of-truth flow).
app.post("/admin/sync", async (req, res) => {
  if (!adminOK(req)) return res.sendStatus(403);
  await syncPaidFromStripe();
  res.json({ ok: true, paid: [...paidNumbers] });
});

// Stripe webhook — flips a number to paid/unpaid when a subscription starts or
// ends. Ready for when Stripe is configured (STRIPE_WEBHOOK_SECRET).
app.post("/stripe/webhook", (req, res) => {
  let event;
  try {
    event = STRIPE_WEBHOOK_SECRET
      ? verifyStripe(req.body, req.headers["stripe-signature"], STRIPE_WEBHOOK_SECRET)
      : JSON.parse(req.body.toString("utf8"));
  } catch (err) {
    console.warn("stripe verify failed:", err.message);
    return res.sendStatus(400);
  }
  try {
    const o = event.data?.object || {};
    const phone = normalizePhone(o.client_reference_id || o.customer_details?.phone || o.metadata?.phone || o.phone || "");
    const start = ["checkout.session.completed", "invoice.paid", "customer.subscription.created"];
    const end = ["customer.subscription.deleted", "invoice.payment_failed"];
    if (phone && phone !== "+") {
      if (start.includes(event.type)) {
        setPaid(phone, true);
        console.log("✅ subscribed:", phone);
        verifyContact(phone); // make sure we can text them (reminders, etc.)
        // Stamp the phone on the Stripe customer so restarts can re-sync from Stripe.
        if (o.customer && STRIPE_SECRET_KEY) {
          stripeApi("customers/" + o.customer, { method: "POST", body: new URLSearchParams({ "metadata[phone]": phone }).toString() }).catch(() => {});
        }
      } else if (end.includes(event.type)) {
        setPaid(phone, false);
        console.log("⛔ unsubscribed:", phone);
      }
    }
  } catch (err) {
    console.error("stripe handler error:", err.message);
  }
  res.json({ received: true });
});

// Heartbeat target: a scheduler (or our own self-ping) hits this every few minutes
// to keep the server awake and fire any due reminders. Safe to call publicly — it
// only sends reminders the user already scheduled.
app.all("/cron/tick", async (_req, res) => {
  try {
    const result = await runDueReminders();
    res.json({ ok: true, ...result });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

// Simple health check.
app.get("/health", (_req, res) => res.json({ ok: true, reminders: reminders.length }));

app.listen(PORT, () => {
  console.log(`Rehearse running on port ${PORT}`);
  console.log(`Model: ${GROQ_MODEL} via Groq`);
  console.log(`Webhook: POST /webhook/sendblue   Test: POST /test   Tick: ALL /cron/tick`);
  syncPaidFromStripe(); // load current subscribers from Stripe on boot
});

/**
 * Self-driven scheduler + keep-alive. Every few minutes we fire any due reminders
 * and ping our own public URL. On Render's free tier that self-ping is the
 * external traffic needed to avoid the 15-minute idle sleep, so the coach stays
 * awake to send proactive texts — no external cron or GitHub Action required.
 * (RENDER_EXTERNAL_URL is injected automatically by Render.)
 */
const SELF_URL = process.env.RENDER_EXTERNAL_URL;
const TICK_MS = 5 * 60 * 1000;
setInterval(async () => {
  try { await runDueReminders(); } catch (err) { console.error("scheduler tick error:", err.message); }
  try { await syncPaidFromStripe(); } catch (err) { console.error("stripe sync tick error:", err.message); }
  try { await cleanupStaleLeads(); } catch (err) { console.error("lead cleanup tick error:", err.message); }
  if (SELF_URL) { try { await fetch(`${SELF_URL}/health`); } catch { /* ignore */ } }
}, TICK_MS);
