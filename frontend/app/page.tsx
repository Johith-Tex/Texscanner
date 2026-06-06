"use client";
import { useState, useRef } from "react";
import {
  Shield, ShieldAlert, Zap, Mail, BarChart2, AlertTriangle,
  CheckCircle, XCircle, ChevronDown, Copy, RotateCcw, Cpu,
  Eye, TrendingUp, Hash, ExternalLink
} from "lucide-react";

// ── Types ──────────────────────────────────────────────────────────────────
interface ConfidenceData {
  spam: number;
  ham: number;
  level: string;
}

interface AnalysisData {
  spam_keywords: string[];
  url_count: number;
  exclamation_marks: number;
  question_marks: number;
  dollar_signs: number;
  caps_words: string[];
  caps_ratio: number;
  pattern_matches: string[];
  word_count: number;
  char_count: number;
}

interface ClassifyResult {
  prediction: "spam" | "ham";
  is_spam: boolean;
  confidence: ConfidenceData;
  model_used: string;
  analysis: AnalysisData;
}

// ── Constants ──────────────────────────────────────────────────────────────
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

const SAMPLE_EMAILS = {
  spam: `Congratulations!!! You've WON $1,000,000 in our EXCLUSIVE lottery! 
  
URGENT: Click HERE NOW to claim your PRIZE before it expires in 24 hours! This LIMITED TIME OFFER is only for selected winners like you!

FREE gift included! No purchase necessary! Act NOW!!!
  
To unsubscribe click here.`,
  ham: `Hi Sarah,

I wanted to follow up on our conversation from last Tuesday regarding the Q3 budget proposal. 

After reviewing the numbers with the finance team, we're comfortable moving forward with the revised estimate. Could you please schedule a call this week to finalize the details?

Thanks for your patience on this.

Best,
Michael`,
};

// ── Sub-Components ─────────────────────────────────────────────────────────
function StatBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs" style={{ color: "var(--muted)" }}>
        <span>{label}</span>
        <span style={{ color }}>{value.toFixed(1)}%</span>
      </div>
      <div className="h-1.5 rounded-full overflow-hidden" style={{ background: "var(--border)" }}>
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${value}%`, background: color }}
        />
      </div>
    </div>
  );
}

function Chip({ label, color }: { label: string; color?: string }) {
  return (
    <span
      className="px-2 py-0.5 rounded text-xs font-mono"
      style={{
        background: color ? `${color}20` : "var(--surface-2)",
        color: color || "var(--muted)",
        border: `1px solid ${color ? `${color}40` : "var(--border)"}`,
      }}
    >
      {label}
    </span>
  );
}

function MetricCard({ icon: Icon, label, value, sub }: {
  icon: React.ElementType; label: string; value: string | number; sub?: string;
}) {
  return (
    <div className="glass rounded-xl p-4 flex items-center gap-3">
      <div className="p-2 rounded-lg" style={{ background: "var(--surface-2)" }}>
        <Icon size={16} style={{ color: "var(--accent)" }} />
      </div>
      <div>
        <div className="text-xs" style={{ color: "var(--muted)" }}>{label}</div>
        <div className="font-semibold text-sm">{value}</div>
        {sub && <div className="text-xs" style={{ color: "var(--muted)" }}>{sub}</div>}
      </div>
    </div>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────────
export default function Home() {
  const [text, setText] = useState("");
  const [model, setModel] = useState("best");
  const [result, setResult] = useState<ClassifyResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAnalysis, setShowAnalysis] = useState(false);
  const [copied, setCopied] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  async function classify() {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setShowAnalysis(false);

    try {
      const res = await fetch(`${API_BASE}/classify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, model, analysis: true }),
      });
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to connect to API";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setText("");
    setResult(null);
    setError(null);
    setShowAnalysis(false);
    textareaRef.current?.focus();
  }

  function copyResult() {
    if (!result) return;
    navigator.clipboard.writeText(JSON.stringify(result, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  const isSpam = result?.is_spam;
  const confidence = result?.confidence;
  const analysis = result?.analysis;

  return (
    <div className="min-h-screen" style={{ background: "var(--bg)" }}>
      {/* Scanner line effect */}
      <div className="scanner-line" />

      {/* Background grid */}
      <div
        className="fixed inset-0 pointer-events-none opacity-30"
        style={{
          backgroundImage: `linear-gradient(var(--border) 1px, transparent 1px),
            linear-gradient(90deg, var(--border) 1px, transparent 1px)`,
          backgroundSize: "60px 60px",
        }}
      />

      {/* Ambient glow */}
      <div
        className="fixed top-0 left-1/2 -translate-x-1/2 w-[600px] h-[400px] pointer-events-none"
        style={{
          background: "radial-gradient(ellipse, rgba(108,99,255,0.12) 0%, transparent 70%)",
        }}
      />

      <div className="relative max-w-5xl mx-auto px-4 py-12">
        {/* Header */}
        <header className="text-center mb-14">
          <div className="flex items-center justify-center gap-3 mb-5">
            <div
              className="p-3 rounded-2xl glow-accent"
              style={{ background: "var(--surface)", border: "1px solid var(--accent)" }}
            >
              <Shield size={28} style={{ color: "var(--accent)" }} />
            </div>
            <h1 className="text-5xl font-bold tracking-tight gradient-text">TexScanner</h1>
          </div>
          <p className="text-lg max-w-lg mx-auto" style={{ color: "var(--muted)" }}>
            Production-grade spam detection powered by{" "}
            <span style={{ color: "var(--text)" }}>Naive Bayes</span> &{" "}
            <span style={{ color: "var(--text)" }}>SVM</span> classifiers
          </p>

          {/* Model selector */}
          <div className="flex items-center justify-center gap-2 mt-6">
            {["best", "svm", "naive_bayes"].map((m) => (
              <button
                key={m}
                onClick={() => setModel(m)}
                className="px-4 py-1.5 rounded-full text-sm font-medium transition-all duration-200"
                style={{
                  background: model === m ? "var(--accent)" : "var(--surface-2)",
                  color: model === m ? "#fff" : "var(--muted)",
                  border: `1px solid ${model === m ? "var(--accent)" : "var(--border)"}`,
                }}
              >
                {m === "best" ? "⚡ Best" : m === "svm" ? "◆ SVM" : "◉ Naïve Bayes"}
              </button>
            ))}
          </div>
        </header>

        {/* Input Section */}
        <div className="glass rounded-2xl p-6 mb-6" style={{ borderColor: "var(--border)" }}>
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Mail size={16} style={{ color: "var(--accent)" }} />
              <span className="text-sm font-medium">Email Content</span>
            </div>
            <div className="flex gap-2">
              {Object.entries(SAMPLE_EMAILS).map(([key, val]) => (
                <button
                  key={key}
                  onClick={() => { setText(val); setResult(null); }}
                  className="text-xs px-3 py-1 rounded-full transition-all"
                  style={{
                    background: "var(--surface-2)",
                    color: key === "spam" ? "var(--spam)" : "var(--ham)",
                    border: `1px solid ${key === "spam" ? "rgba(255,71,87,0.3)" : "rgba(46,213,115,0.3)"}`,
                  }}
                >
                  {key === "spam" ? "🚫 Spam sample" : "✅ Ham sample"}
                </button>
              ))}
            </div>
          </div>

          <textarea
            ref={textareaRef}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste email content here to analyze..."
            className="w-full h-44 text-sm resize-none rounded-xl p-4 transition-all duration-200"
            style={{
              background: "var(--surface-2)",
              color: "var(--text)",
              border: "1px solid var(--border)",
              fontFamily: "DM Mono, monospace",
              lineHeight: 1.6,
            }}
          />

          <div className="flex items-center justify-between mt-4">
            <span className="text-xs" style={{ color: "var(--muted)" }}>
              {text.length} chars · {text.trim().split(/\s+/).filter(Boolean).length} words
            </span>
            <div className="flex gap-3">
              {(result || text) && (
                <button
                  onClick={reset}
                  className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm transition-all"
                  style={{ color: "var(--muted)", background: "var(--surface-2)", border: "1px solid var(--border)" }}
                >
                  <RotateCcw size={14} />
                  Reset
                </button>
              )}
              <button
                onClick={classify}
                disabled={!text.trim() || loading}
                className="flex items-center gap-2 px-6 py-2 rounded-xl text-sm font-semibold transition-all duration-200"
                style={{
                  background: loading || !text.trim()
                    ? "var(--surface-2)"
                    : "linear-gradient(135deg, var(--accent), #9b8ff0)",
                  color: loading || !text.trim() ? "var(--muted)" : "#fff",
                  border: "none",
                  cursor: loading || !text.trim() ? "not-allowed" : "pointer",
                }}
              >
                {loading ? (
                  <>
                    <div
                      className="w-4 h-4 rounded-full border-2 border-t-transparent animate-spin"
                      style={{ borderColor: "var(--accent)", borderTopColor: "transparent" }}
                    />
                    Scanning...
                  </>
                ) : (
                  <>
                    <Zap size={16} />
                    Classify
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div
            className="glass rounded-xl p-4 mb-6 flex items-start gap-3"
            style={{ borderColor: "rgba(255,71,87,0.4)", background: "rgba(255,71,87,0.05)" }}
          >
            <AlertTriangle size={18} style={{ color: "var(--spam)", flexShrink: 0, marginTop: 1 }} />
            <div>
              <div className="text-sm font-medium" style={{ color: "var(--spam)" }}>Connection Error</div>
              <div className="text-sm mt-1" style={{ color: "var(--muted)" }}>{error}</div>
              <div className="text-xs mt-2" style={{ color: "var(--muted)" }}>
                Make sure the Flask API is running at <code style={{ color: "var(--accent)" }}>{API_BASE}</code>
              </div>
            </div>
          </div>
        )}

        {/* Result */}
        {result && (
          <div className="space-y-4">
            {/* Verdict banner */}
            <div
              className="rounded-2xl p-6 relative overflow-hidden"
              style={{
                border: `2px solid ${isSpam ? "rgba(255,71,87,0.5)" : "rgba(46,213,115,0.5)"}`,
                background: isSpam
                  ? "linear-gradient(135deg, rgba(255,71,87,0.08), rgba(255,71,87,0.02))"
                  : "linear-gradient(135deg, rgba(46,213,115,0.08), rgba(46,213,115,0.02))",
              }}
            >
              <div
                className="absolute inset-0 opacity-5"
                style={{
                  background: `radial-gradient(circle at 0% 50%, ${isSpam ? "#ff4757" : "#2ed573"} 0%, transparent 60%)`,
                }}
              />
              <div className="relative flex items-center justify-between flex-wrap gap-4">
                <div className="flex items-center gap-4">
                  <div
                    className="p-4 rounded-2xl"
                    style={{ background: isSpam ? "rgba(255,71,87,0.15)" : "rgba(46,213,115,0.15)" }}
                  >
                    {isSpam
                      ? <ShieldAlert size={32} style={{ color: "var(--spam)" }} />
                      : <CheckCircle size={32} style={{ color: "var(--ham)" }} />
                    }
                  </div>
                  <div>
                    <div className="text-3xl font-bold" style={{ color: isSpam ? "var(--spam)" : "var(--ham)" }}>
                      {isSpam ? "SPAM DETECTED" : "CLEAN EMAIL"}
                    </div>
                    <div className="text-sm mt-1" style={{ color: "var(--muted)" }}>
                      {confidence?.level} confidence · {result.model_used} model
                    </div>
                  </div>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={copyResult}
                    className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs transition-all"
                    style={{ background: "var(--surface-2)", color: "var(--muted)", border: "1px solid var(--border)" }}
                  >
                    <Copy size={12} />
                    {copied ? "Copied!" : "Copy JSON"}
                  </button>
                </div>
              </div>

              {/* Confidence bars */}
              <div className="mt-6 grid grid-cols-2 gap-4">
                <StatBar label="Spam probability" value={confidence?.spam ?? 0} color="var(--spam)" />
                <StatBar label="Ham probability" value={confidence?.ham ?? 0} color="var(--ham)" />
              </div>
            </div>

            {/* Metrics grid */}
            {analysis && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <MetricCard icon={Hash} label="Word count" value={analysis.word_count} />
                <MetricCard
                  icon={AlertTriangle}
                  label="Spam signals"
                  value={analysis.pattern_matches.length}
                  sub={analysis.pattern_matches.length > 0 ? "patterns matched" : "none found"}
                />
                <MetricCard
                  icon={TrendingUp}
                  label="CAPS ratio"
                  value={`${analysis.caps_ratio}%`}
                  sub={analysis.caps_ratio > 20 ? "high" : "normal"}
                />
                <MetricCard
                  icon={ExternalLink}
                  label="URLs found"
                  value={analysis.url_count}
                />
              </div>
            )}

            {/* Analysis accordion */}
            {analysis && (
              <div className="glass rounded-2xl overflow-hidden">
                <button
                  onClick={() => setShowAnalysis(!showAnalysis)}
                  className="w-full flex items-center justify-between p-5 transition-colors hover:bg-white/5"
                >
                  <div className="flex items-center gap-2">
                    <Eye size={16} style={{ color: "var(--accent)" }} />
                    <span className="font-medium text-sm">Detailed Analysis</span>
                  </div>
                  <ChevronDown
                    size={16}
                    style={{
                      color: "var(--muted)",
                      transform: showAnalysis ? "rotate(180deg)" : "none",
                      transition: "transform 0.2s",
                    }}
                  />
                </button>

                {showAnalysis && (
                  <div className="p-5 pt-0 space-y-5" style={{ borderTop: "1px solid var(--border)" }}>

                    {analysis.pattern_matches.length > 0 && (
                      <div>
                        <div className="text-xs font-medium mb-2" style={{ color: "var(--muted)" }}>
                          TRIGGERED PATTERNS
                        </div>
                        <div className="flex flex-wrap gap-2">
                          {analysis.pattern_matches.map((p, i) => (
                            <Chip key={i} label={p} color="var(--spam)" />
                          ))}
                        </div>
                      </div>
                    )}

                    {analysis.spam_keywords.length > 0 && (
                      <div>
                        <div className="text-xs font-medium mb-2" style={{ color: "var(--muted)" }}>
                          SPAM KEYWORDS FOUND
                        </div>
                        <div className="flex flex-wrap gap-2">
                          {analysis.spam_keywords.map((kw, i) => (
                            <Chip key={i} label={kw} color="#f39c12" />
                          ))}
                        </div>
                      </div>
                    )}

                    <div className="grid grid-cols-3 gap-3 text-center">
                      {[
                        { label: "! marks", value: analysis.exclamation_marks },
                        { label: "? marks", value: analysis.question_marks },
                        { label: "$ signs", value: analysis.dollar_signs },
                      ].map(({ label, value }) => (
                        <div key={label} className="rounded-xl p-3" style={{ background: "var(--surface-2)" }}>
                          <div className="text-xl font-bold">{value}</div>
                          <div className="text-xs" style={{ color: "var(--muted)" }}>{label}</div>
                        </div>
                      ))}
                    </div>

                    {analysis.caps_words.length > 0 && (
                      <div>
                        <div className="text-xs font-medium mb-2" style={{ color: "var(--muted)" }}>
                          ALL-CAPS WORDS
                        </div>
                        <div className="flex flex-wrap gap-2">
                          {analysis.caps_words.map((w, i) => (
                            <Chip key={i} label={w} />
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Footer info */}
        <footer className="mt-16 text-center space-y-2">
          <div className="flex items-center justify-center gap-6 text-xs" style={{ color: "var(--muted)" }}>
            <div className="flex items-center gap-1.5"><Cpu size={12} /> Naive Bayes + SVM</div>
            <div className="flex items-center gap-1.5"><BarChart2 size={12} /> TF-IDF Features</div>
            <div className="flex items-center gap-1.5"><Shield size={12} /> TexScanner v1.0</div>
          </div>
          <div className="text-xs" style={{ color: "var(--muted)" }}>
            Deploy on GitHub + Vercel · Flask API + Next.js
          </div>
        </footer>
      </div>
    </div>
  );
}
