import express from "express";
import dotenv from "dotenv";
import crypto from "crypto";

dotenv.config();

const {
  SENDBLUE_API_KEY,
  SENDBLUE_API_SECRET,
  SENDBLUE_FROM_NUMBER, // optional: your Sendblue line to send from (e.g. +1...)
  GROQ_API_KEY,
  GROQ_MODEL = "llama-3.1-8b-instant", // sales + background tasks (cheap/fast)
  PAID_MODEL = "llama-3.3-70b-versatile", // the actual coaching for subscribers (higher quality)
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
app.use(express.static(".", { extensions: ["html"] })); // serves index.html + /terms, /privacy

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
- If they say "new", "restart", or name a different company, switch gracefully.
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
- Do NOT invent or promise a free trial, discount, or refund. The ONLY offer is the ${PRICE} subscription (cancel anytime).

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
const paidNumbers = new Set();   // everyone with access now (Stripe-active ∪ manual grants)
const manualGrants = new Set();  // admin grants, not backed by Stripe (survive authoritative sync)
const customerId = new Map();    // phone -> Stripe customer id (to persist per-user state)
const lazyChecked = new Map();   // phone -> last Stripe lookup ts (negative cache)
const optedOut = new Set();      // numbers who texted STOP — never message them until START
let stateLoaded = false;         // have we rehydrated reminders from Stripe yet?

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

/** Re-add a persisted reminder on boot (skipping one-offs whose time has passed). */
function rehydrateReminder(r) {
  if (!r || typeof r !== "object") return;
  if (r.kind === "once" && (!r.fireAt || r.fireAt <= Date.now())) return;
  reminders.push({ ...r, id: reminderSeq++ });
}

/**
 * Authoritatively rebuild the paid list from Stripe (the source of truth) plus any
 * manual admin grants. Runs on startup and every few minutes, so paid customers
 * stay unlocked across restarts and canceled ones are dropped even if a webhook
 * was missed. On the first run it also rehydrates each subscriber's reminders +
 * timezone from their Stripe customer metadata.
 */
async function syncPaidFromStripe() {
  if (!STRIPE_SECRET_KEY) return;
  try {
    const params = new URLSearchParams({ status: "active", limit: "100" });
    params.append("expand[]", "data.customer");
    const subs = await stripeApi("subscriptions?" + params.toString());
    const active = new Set();
    for (const sub of subs.data || []) {
      const c = sub.customer || {};
      const phone = normalizePhone(c.metadata?.phone || c.phone || "");
      if (!phone || phone === "+") continue;
      active.add(phone);
      customerId.set(phone, c.id);
      if (!stateLoaded) {
        if (c.metadata?.tz && isValidTz(c.metadata.tz)) timezones.set(phone, c.metadata.tz);
        if (c.metadata?.optedOut === "1") optedOut.add(phone);
        if (c.metadata?.reminders) {
          try { for (const r of JSON.parse(c.metadata.reminders)) rehydrateReminder(r); } catch { /* ignore bad blob */ }
        }
      }
    }
    // Authoritative: access = currently-active Stripe subs ∪ manual grants.
    paidNumbers.clear();
    for (const p of active) paidNumbers.add(p);
    for (const p of manualGrants) paidNumbers.add(p);
    stateLoaded = true;
    console.log(`Stripe sync: ${active.size} active subscriber(s)`);
  } catch (err) {
    console.warn("Stripe sync failed:", err.message);
  }
}

/** Persist a paid user's reminders + timezone into their Stripe customer metadata. */
async function saveCustomerState(number) {
  const cid = customerId.get(number);
  if (!cid || !STRIPE_SECRET_KEY) return;
  let mine = reminders.filter((r) => r.number === number).map(({ id, ...rest }) => rest);
  let blob = JSON.stringify(mine);
  while (blob.length > 480 && mine.length) { mine.shift(); blob = JSON.stringify(mine); } // Stripe metadata caps ~500 chars
  const body = new URLSearchParams();
  body.set("metadata[reminders]", blob);
  body.set("metadata[tz]", tzFor(number));
  body.set("metadata[optedOut]", optedOut.has(number) ? "1" : "");
  stripeApi("customers/" + cid, { method: "POST", body: body.toString() }).catch((e) => console.warn("save state:", e.message));
}

/**
 * For an unknown caller, ask Stripe directly whether they have an active
 * subscription (covers the gap right after a restart, and people who pay on the
 * website then text in). Negative results are cached ~10 min to avoid hammering.
 */
async function lazyStripeCheck(number) {
  if (!STRIPE_SECRET_KEY) return false;
  const last = lazyChecked.get(number) || 0;
  if (Date.now() - last < 10 * 60 * 1000) return false;
  lazyChecked.set(number, Date.now());
  try {
    const q = encodeURIComponent(`metadata['phone']:'${number}'`);
    const found = await stripeApi(`customers/search?query=${q}&limit=1`);
    const cust = found.data?.[0];
    if (!cust) return false;
    const subs = await stripeApi(`subscriptions?customer=${cust.id}&status=active&limit=1`);
    if (subs.data?.length) {
      setPaid(number, true);
      customerId.set(number, cust.id);
      return true;
    }
  } catch (err) {
    console.warn("lazy check:", err.message);
  }
  return false;
}

const SITE_URL = process.env.RENDER_EXTERNAL_URL || "https://rehearse-143f.onrender.com";

/** Does this message look like a billing / cancellation request (not "cancel reminders")? */
function cancelIntent(text) {
  return /\b(unsubscribe|cancel\s+(my\s+)?(subscription|membership|plan|account|billing)|manage\s+(my\s+)?(subscription|billing|plan|account)|stop\s+(my\s+)?(subscription|membership)|update\s+(my\s+)?(card|payment)|refund|billing\s+(help|issue|portal|question))\b/i.test(text || "");
}

// ---- Compliance: STOP / START / HELP keyword handling -------------------------
const STOP_CONFIRM = "You're unsubscribed and won't receive more messages from Rehearse. Reply START anytime to resume.";
const START_CONFIRM = "You're resubscribed — welcome back. Text a company or role to practice anytime.";
const HELP_REPLY = `Rehearse — your AI interview coach over text (${PRICE}). Reply START to resume, STOP to opt out. Questions: niah@changeist.org`;

function optOutIntent(text) {
  const t = (text || "").trim().toLowerCase();
  if (["stop", "stopall", "stop all", "unsubscribe", "unsubscribe me", "optout", "opt out", "opt-out"].includes(t)) return true;
  return /\b(stop\s+(texting|messaging|contacting)(\s+me)?|unsubscribe\s+me|remove\s+me|leave\s+me\s+alone|do\s*n'?t\s+(text|message|contact)\s+me|do not\s+(text|message|contact)\s+me)\b/i.test(text || "");
}
function resumeIntent(text) {
  const t = (text || "").trim().toLowerCase();
  if (["start", "unstop", "resume", "opt in", "opt-in", "optin"].includes(t)) return true;
  return /\b(start\s+(texting|messaging)|opt\s+back\s+in|re-?subscribe)\b/i.test(text || "");
}
function helpIntent(text) {
  return ["help", "info"].includes((text || "").trim().toLowerCase());
}

/** Create a Stripe Customer Portal link so a subscriber can manage/cancel themselves. */
async function billingPortalUrl(number) {
  const cid = customerId.get(number);
  if (!cid || !STRIPE_SECRET_KEY) return null;
  try {
    const body = new URLSearchParams({ customer: cid, return_url: SITE_URL });
    const session = await stripeApi("billing_portal/sessions", { method: "POST", body: body.toString() });
    return session.url || null;
  } catch (err) {
    console.warn("portal session:", err.message);
    return null;
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
async function askModel(history, systemPrompt, { temperature = 0.8, maxTokens = 300, model = GROQ_MODEL } = {}) {
  const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${GROQ_API_KEY}`,
    },
    body: JSON.stringify({
      model,
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

const lastSeen = new Map();       // number -> last inbound timestamp (ms)
const leadLine = new Map();       // number -> Sendblue line that received their texts
const finalPitchSent = new Set(); // numbers we've sent the single win-back to
const LEAD_TTL_MS = (Number(LEAD_TTL_HOURS) || 24) * 3600e3;
const WINBACK_MS = (Number(process.env.WINBACK_MINUTES) || 30) * 60000;

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
    leadLine.delete(number);
    finalPitchSent.delete(number);
    console.log("🧹 removed stale unpaid lead:", number);
  }
}

/** A warm, final win-back pitch for an unpaid lead who went quiet. */
async function winBackMessage(number) {
  // Build the personalized link in code (the model tends to drop query params).
  const link = `${SUBSCRIBE_URL}${SUBSCRIBE_URL.includes("?") ? "&" : "?"}client_reference_id=${String(number).replace(/\D/g, "")}`;
  const sys = `You are Rehearse, a professional AI interview coach. This person texted you earlier but didn't subscribe. Write ONE short, warm, confident final message (2-3 sentences) that gently wins them back: remind them what they're missing (realistic mock interviews, instant feedback, prep reminders before the real thing), note it's ${PRICE}, unlimited, cancel anytime, and invite them to start. Reference their situation if the conversation shows it. Do NOT include any link or URL — one is appended automatically. Plain text only, no markdown.`;
  let msg;
  try {
    const convo = getHistory(number).slice(-6).map((m) => `${m.role === "user" ? "User" : "Coach"}: ${m.content}`).join("\n");
    msg = await askModel([{ role: "user", content: `Earlier conversation:\n${convo || "(brief)"}\n\nWrite the final win-back message now.` }], sys, { temperature: 0.7, maxTokens: 160 });
  } catch {
    msg = `Still on the fence? Rehearse is your AI interview coach right in your texts — unlimited mock interviews, instant feedback, and reminders before the real thing. ${PRICE}, cancel anytime.`;
  }
  return `${msg.trim()}\n\n${link}`;
}

/** Send the single final win-back to unpaid leads who went quiet ~30 min ago. */
async function sendWinBacks() {
  const now = Date.now();
  for (const [number, ts] of lastSeen) {
    if (optedOut.has(number)) continue;       // they opted out — never pitch
    if (isPaid(number)) continue;             // they bought
    if (customerId.has(number)) continue;     // existing/former Stripe customer — don't pitch
    if (finalPitchSent.has(number)) continue; // only ever send one
    if (now - ts < WINBACK_MS) continue;      // not quiet long enough yet
    finalPitchSent.add(number);
    try {
      const line = leadLine.get(number) || SENDBLUE_FROM_NUMBER;
      const msg = await winBackMessage(number);
      await sbNotify("send-typing-indicator", number, line);
      await sendText(number, msg, line);
      getHistory(number).push({ role: "assistant", content: msg });
      console.log(`🪝 win-back → ${number}`);
    } catch (err) {
      console.warn("win-back error:", err.message);
    }
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
  saveCustomerState(number); // persist reminder/timezone changes to Stripe
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

/** Welcome a brand-new subscriber so they don't have to figure out the next step. */
async function sendWelcome(number) {
  if (optedOut.has(number)) return;
  const line = leadLine.get(number) || SENDBLUE_FROM_NUMBER;
  const msg = "Welcome to Rehearse 🎉 You're all set. Text me any company or role — like \"Amazon\" or \"product manager\" — and I'll start your first mock interview. Want a reminder before your real interview? Just tell me the date.";
  try {
    await sendText(number, msg, line);
    getHistory(number).push({ role: "assistant", content: msg });
    console.log("👋 welcome →", number);
  } catch (err) {
    // Expected for pay-first buyers who never texted in (unverified) — the
    // Stripe success page covers them instead.
    console.warn("welcome send skipped:", err.message);
  }
}

/** Send a proactive (coach-initiated) text and record it in the conversation. */
async function deliverProactive(r, message) {
  if (optedOut.has(r.number)) return; // never message someone who opted out
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
      "You are an interview coach. Reply with a single short interview question in plain text, no preamble.",
      { model: PAID_MODEL } // daily questions are a paid feature — use the better model
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
  const changed = new Set();
  for (let i = reminders.length - 1; i >= 0; i--) {
    const r = reminders[i];
    try {
      if (r.kind === "once" && r.fireAt <= now) {
        await deliverProactive(r, r.note);
        reminders.splice(i, 1);
        changed.add(r.number);
        sent++;
      } else if (r.kind === "daily") {
        const { date, hhmm } = nowInTz(tzFor(r.number));
        if (hhmm >= r.recurAt && r.lastSent !== date) {
          // Only mark "sent today" if the send actually succeeds, so a failure retries next tick.
          await deliverProactive(r, await generateDailyQuestion(r.context));
          r.lastSent = date;
          changed.add(r.number);
          sent++;
        }
      }
    } catch (err) {
      console.error("reminder fire error:", err.message);
    }
  }
  for (const number of changed) saveCustomerState(number); // persist fired/updated reminders
  return { sent, pending: reminders.length };
}

// Simple per-number rate limit so a single sender can't run up Groq/Sendblue cost.
const rateWindow = new Map(); // number -> recent message timestamps
function rateLimited(number) {
  const now = Date.now();
  const recent = (rateWindow.get(number) || []).filter((t) => now - t < 60000);
  recent.push(now);
  rateWindow.set(number, recent);
  return recent.length > 12; // >12 messages/minute from one number
}

/**
 * Sendblue posts every inbound message here.
 * Set this URL as your "Receive URL" in the Sendblue dashboard.
 */
app.post("/webhook/sendblue", async (req, res) => {
  // Acknowledge immediately so Sendblue doesn't retry while we think.
  res.sendStatus(200);

  try {
    const { content, from_number, is_outbound } = req.body || {};

    // Ignore our own outbound messages and empty payloads.
    if (is_outbound || !content || !from_number) return;
    if (rateLimited(from_number)) { console.warn("rate limited:", from_number); return; }

    // Which of our lines received this? Reply from that same line.
    const line = req.body.to_number || req.body.number || SENDBLUE_FROM_NUMBER;

    // Compliance first: honor STOP / START / HELP before any other handling.
    if (optOutIntent(content)) {
      optedOut.add(from_number);
      if (customerId.has(from_number)) saveCustomerState(from_number); // persist for subscribers
      await sendText(from_number, STOP_CONFIRM, line);
      console.log(`⛔ opt-out: ${from_number}`);
      return;
    }
    if (resumeIntent(content)) {
      optedOut.delete(from_number);
      if (customerId.has(from_number)) saveCustomerState(from_number);
      await sendText(from_number, START_CONFIRM, line);
      console.log(`▶ opt-in: ${from_number}`);
      return;
    }
    if (optedOut.has(from_number)) {
      if (helpIntent(content)) await sendText(from_number, HELP_REPLY, line);
      return; // opted out → stay silent
    }
    if (helpIntent(content)) { await sendText(from_number, HELP_REPLY, line); return; }

    console.log(`← ${from_number} (${content.length} chars)`);
    lastSeen.set(from_number, Date.now()); // track activity for lead cleanup
    leadLine.set(from_number, line);       // remember the line for proactive win-backs

    // Mark their message "Read" and show the typing "…" bubble while we think.
    await sbNotify("mark-read", from_number, line);
    await sbNotify("send-typing-indicator", from_number, line);

    // Paywall: subscribers get full coaching; everyone else gets the sales flow.
    let paid = isPaid(from_number);
    if (!paid) paid = await lazyStripeCheck(from_number); // catch website / post-restart payers
    const history = getHistory(from_number);
    history.push({ role: "user", content });

    // Billing/cancellation requests from subscribers get a self-serve Stripe portal link.
    if (paid && cancelIntent(content)) {
      const url = await billingPortalUrl(from_number);
      const reply = url
        ? `You can manage or cancel your subscription anytime here:\n${url}`
        : `I can help with that — please email niah@changeist.org and we'll take care of it.`;
      history.push({ role: "assistant", content: reply });
      await sendText(from_number, reply, line);
      console.log(`→ ${from_number} (billing portal)`);
      return;
    }

    const systemPrompt = paid ? buildSystemPrompt(from_number) : buildSalesPrompt(from_number);
    const reply = stripDirectives(await askModel(history, systemPrompt, paid ? { model: PAID_MODEL } : {}));
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
    console.log(`→ ${from_number} (${paid ? "paid" : "free"})`);

    // Scheduling is a paid feature — only run the extractor for subscribers.
    if (paid) await extractAndApply(from_number, line, history);
  } catch (err) {
    console.error("webhook error:", err.message);
  }
});

// Local test endpoint: POST {"number":"+1...","content":"Apple"} to simulate an
// inbound text without going through Sendblue. Replies in the JSON response.
app.post("/test", async (req, res) => {
  if (!adminOK(req)) return res.sendStatus(403); // not public — it can bypass the paywall
  try {
    const { number = "+10000000000", content, line, paid: paidFlag } = req.body || {};
    if (!content) return res.status(400).json({ error: "content required" });
    // Mirror the webhook's compliance handling so it can be tested.
    if (optOutIntent(content)) { optedOut.add(number); if (customerId.has(number)) saveCustomerState(number); return res.json({ optedOut: true, reply: STOP_CONFIRM }); }
    if (resumeIntent(content)) { optedOut.delete(number); if (customerId.has(number)) saveCustomerState(number); return res.json({ optedOut: false, reply: START_CONFIRM }); }
    if (optedOut.has(number)) return res.json({ suppressed: true, reply: helpIntent(content) ? HELP_REPLY : null });
    const paid = paidFlag !== undefined ? !!paidFlag : (isPaid(number) || await lazyStripeCheck(number));
    const history = getHistory(number);
    history.push({ role: "user", content });
    if (paid && cancelIntent(content)) {
      const url = await billingPortalUrl(number);
      const reply = url ? `Manage or cancel here: ${url}` : `Contact support to manage your subscription.`;
      history.push({ role: "assistant", content: reply });
      return res.json({ reply, paid, billing: true });
    }
    const reply = stripDirectives(await askModel(history, paid ? buildSystemPrompt(number) : buildSalesPrompt(number), paid ? { model: PAID_MODEL } : {}));
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
  manualGrants.add(n);
  setPaid(n, true);
  res.json({ ok: true, number: n, paid: true });
});
app.post("/admin/revoke", (req, res) => {
  if (!adminOK(req)) return res.sendStatus(403);
  const n = normalizePhone(req.query.number || req.body?.number);
  if (!n) return res.status(400).json({ error: "number required" });
  manualGrants.delete(n);
  setPaid(n, false);
  res.json({ ok: true, number: n, paid: false });
});
app.get("/admin/status", (req, res) => {
  if (!adminOK(req)) return res.sendStatus(403);
  res.json({
    paid: [...paidNumbers],
    stripeKeyConfigured: !!STRIPE_SECRET_KEY,
    webhookSecretConfigured: !!STRIPE_WEBHOOK_SECRET,
    subscribeUrl: SUBSCRIBE_URL,
  });
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
    // Unlock only when payment is confirmed; revoke only on a real cancellation
    // (NOT on a single failed charge — Stripe retries those for days).
    const start = ["checkout.session.completed", "invoice.paid"];
    const end = ["customer.subscription.deleted"];
    if (phone && phone !== "+") {
      if (start.includes(event.type)) {
        const isNew = !isPaid(phone);
        setPaid(phone, true);
        if (o.customer) customerId.set(phone, o.customer);
        console.log("✅ subscribed:", phone);
        verifyContact(phone); // make sure we can text them (reminders, etc.)
        // Stamp the phone on the Stripe customer so restarts can re-sync from Stripe.
        if (o.customer && STRIPE_SECRET_KEY) {
          stripeApi("customers/" + o.customer, { method: "POST", body: new URLSearchParams({ "metadata[phone]": phone }).toString() }).catch(() => {});
        }
        // Welcome them once, only on the initial checkout (not on monthly renewals).
        if (event.type === "checkout.session.completed" && isNew) sendWelcome(phone);
      } else if (end.includes(event.type)) {
        manualGrants.delete(phone);
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
  try { await sendWinBacks(); } catch (err) { console.error("win-back tick error:", err.message); }
  try { await cleanupStaleLeads(); } catch (err) { console.error("lead cleanup tick error:", err.message); }
  if (SELF_URL) { try { await fetch(`${SELF_URL}/health`); } catch { /* ignore */ } }
}, TICK_MS);
