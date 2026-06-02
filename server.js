import express from "express";
import dotenv from "dotenv";

dotenv.config();

const {
  SENDBLUE_API_KEY,
  SENDBLUE_API_SECRET,
  SENDBLUE_FROM_NUMBER, // optional: your Sendblue line to send from (e.g. +1...)
  GROQ_API_KEY,
  GROQ_MODEL = "llama-3.1-8b-instant",
  PORT = 3000,
} = process.env;

if (!SENDBLUE_API_KEY || !SENDBLUE_API_SECRET || !GROQ_API_KEY) {
  console.error("Missing env vars. Need SENDBLUE_API_KEY, SENDBLUE_API_SECRET, and GROQ_API_KEY.");
  process.exit(1);
}

const MAX_TURNS = 40; // cap stored history (user+assistant messages) per number

const app = express();
app.use(express.json());
app.use(express.static(".")); // serves index.html (the landing page)

/**
 * The coach's personality + rules. Tuned for texting: short, one question at a
 * time, brief feedback, encouraging. Sent to the model as the system message.
 */
const SYSTEM_PROMPT = `You are Rehearse, a friendly AI interview coach talking to someone over text message (iMessage/SMS).

STYLE:
- Keep every reply SHORT — this is a text conversation, not an essay. Usually 1-4 sentences.
- Warm, encouraging, and direct. No corporate fluff.
- Use plain text only. No markdown, no bullet symbols, no headers.

HOW A SESSION WORKS:
- When the user texts a company name (e.g. "Apple") or a role (e.g. "product manager"), start a mock interview tailored to it.
- First, send a quick one-line intro confirming what you'll practice, then ask the FIRST question.
- Ask ONE interview question at a time. Wait for their answer.
- After each answer, give brief, specific feedback (1-2 sentences: what was strong + one thing to improve), then ask the next question.
- Mix behavioral and role-relevant technical questions appropriate to the company/role.
- If they say things like "stop", "new", "restart", or name a different company, gracefully switch.
- If the very first message is just "hi" or unclear, greet them and ask which company or role they want to practice for.

Keep it feeling like a real, momentum-building coaching session over text.`;

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

/** Call Groq (OpenAI-compatible) with the running history; return the reply text. */
async function askModel(history) {
  const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${GROQ_API_KEY}`,
    },
    body: JSON.stringify({
      model: GROQ_MODEL,
      messages: [{ role: "system", content: SYSTEM_PROMPT }, ...history],
      temperature: 0.8,
      max_tokens: 300,
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

    // Which of our lines received this? Reply from that same line.
    const line = req.body.to_number || req.body.number || SENDBLUE_FROM_NUMBER;

    const history = getHistory(from_number);
    history.push({ role: "user", content });

    const reply = await askModel(history);
    history.push({ role: "assistant", content: reply });

    // Trim old turns to cap memory use.
    if (history.length > MAX_TURNS) {
      history.splice(0, history.length - MAX_TURNS);
    }

    await sendText(from_number, reply, line);
    console.log(`→ ${from_number}: ${reply}`);
  } catch (err) {
    console.error("webhook error:", err.message);
  }
});

// Local test endpoint: POST {"number":"+1...","content":"Apple"} to simulate an
// inbound text without going through Sendblue. Replies in the JSON response.
app.post("/test", async (req, res) => {
  try {
    const { number = "+10000000000", content } = req.body || {};
    if (!content) return res.status(400).json({ error: "content required" });
    const history = getHistory(number);
    history.push({ role: "user", content });
    const reply = await askModel(history);
    history.push({ role: "assistant", content: reply });
    res.json({ reply });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Simple health check.
app.get("/health", (_req, res) => res.json({ ok: true }));

app.listen(PORT, () => {
  console.log(`Rehearse running on port ${PORT}`);
  console.log(`Model: ${GROQ_MODEL} via Groq`);
  console.log(`Webhook: POST /webhook/sendblue   Test: POST /test`);
});
