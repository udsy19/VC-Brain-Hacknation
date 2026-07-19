"use client";

/**
 * Decision input #1: what to do, and what is stopping it.
 *
 * This leads the page because it is the only block that answers the question the reader
 * actually arrived with. Everything below it exists to justify or undermine what is
 * stated here.
 *
 * Two things it must never do:
 *   1. Render a blended score. There is none here — the governing axis is NAMED and its
 *      own score shown, because min-axis is the ranking policy. That is a selection, not
 *      an average.
 *   2. Work around the lock. The recommendation is withheld SERVER-SIDE until the dissent
 *      has been served, and this panel renders the padlock rather than pre-fetching the
 *      unlocked memo. The button below hands off to the same shared state the split view
 *      uses, so unlocking here and unlocking there are one event.
 */

import { TIMEOUT } from "@/lib/api";
import type { Recommendation } from "@/lib/types";
import { AXIS_LABEL, type AxisKey } from "@/lib/types";
import type { MemoDissentState } from "@/lib/useMemoDissent";
import { Busy, ErrorNote, GateBadge, Loading, SourceChip } from "./ui";

/** Human labels for the confidence components. The wire names are snake_case internals. */
const COMPONENT_LABEL: Record<string, string> = {
  governing_axis_confidence: "Governing axis confidence",
  founder_interval: "Founder interval width",
  verified_share: "Share of claims verified",
  gap_pressure: "Open gaps in the memo",
};

const label = (name: string) =>
  COMPONENT_LABEL[name] ?? name.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());

const money = (n: number, currency: string) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
    maximumFractionDigits: 0,
  }).format(n);

/**
 * The binding constraint, stated as a sentence.
 *
 * A confidence of 0.40 on its own is not actionable — it does not tell you what to go and
 * find. The API reports WHICH component held the minimum, so this says that instead: the
 * number, the thing holding it down, and the basis for that thing.
 */
function BindingConstraint({ conf }: { conf: NonNullable<Recommendation["confidence"]> }) {
  const binding =
    conf.components.find((c) => c.name === conf.binding_component) ?? null;

  return (
    <div className="border border-[color:var(--rule)] bg-[color:var(--ink-09)] px-4 py-3">
      <div className="meta text-[color:var(--muted)]">What is holding this back</div>
      {binding ? (
        <>
          <p className="mt-1.5 text-[16px] leading-[1.5] text-[color:var(--figure)]">
            <strong>{label(binding.name)}</strong> is the binding constraint, at{" "}
            <span className="mono">{binding.support?.toFixed(2) ?? "—"}</span>. It caps the
            whole recommendation at{" "}
            <span className="mono">{conf.value.toFixed(2)}</span>.
          </p>
          <p className="mt-1.5 text-[14px] leading-[1.55] text-[color:var(--muted)]">
            {binding.basis}
          </p>
        </>
      ) : (
        <p className="mt-1.5 text-[15px] leading-[1.55] text-[color:var(--muted)]">
          The server did not name a binding component for this confidence value, so there
          is nothing to point at. The number stands without an attributed cause.
        </p>
      )}

      {/* The other inputs sit behind a disclosure: they are the working, not the finding. */}
      {conf.components.length > 1 && (
        <details className="mt-3">
          <summary className="meta cursor-pointer text-[color:var(--accent)]">
            All {conf.components.length} confidence inputs — the minimum, never a mean
          </summary>
          <p className="caption mt-2 max-w-none text-[color:var(--muted)]">{conf.method}</p>
          <ul className="mt-2 space-y-1.5">
            {[...conf.components]
              .sort((a, b) => (a.support ?? 2) - (b.support ?? 2))
              .map((c) => {
                const isBinding = c.name === conf.binding_component;
                return (
                  <li
                    key={c.name}
                    className="flex items-baseline gap-3 border-b border-[color:var(--rule)] pb-1.5 last:border-b-0"
                  >
                    <span
                      className="mono w-12 shrink-0 text-right text-[14px]"
                      style={{ color: isBinding ? "var(--accent)" : "var(--muted)" }}
                    >
                      {c.support === null ? "n/a" : c.support.toFixed(2)}
                    </span>
                    <span className="min-w-0 flex-1">
                      <span
                        className="text-[14px]"
                        style={{
                          color: isBinding ? "var(--accent)" : "var(--figure)",
                          fontWeight: isBinding ? 500 : 400,
                        }}
                      >
                        {label(c.name)}
                        {isBinding && " — binding"}
                      </span>
                      <span className="mt-0.5 block text-[12px] text-[color:var(--muted)]">
                        {c.unit}
                      </span>
                    </span>
                  </li>
                );
              })}
          </ul>
        </details>
      )}
    </div>
  );
}

/** The cheque, or the explicit absence of one. A null amount is never drawn as $0. */
function Cheque({ rec }: { rec: Recommendation }) {
  if (rec.amount_usd === null) {
    return (
      <div>
        <div className="meta text-[color:var(--muted)]">Cheque</div>
        <div className="mt-1 flex items-baseline gap-3">
          <span className="font-[family-name:var(--font-instrument-serif)] text-[52px] leading-[0.85] text-[color:var(--muted)]">
            none
          </span>
        </div>
        <p className="caption mt-1.5 max-w-none text-[color:var(--muted)]">
          Not a smaller cheque — no cheque. This is a final answer at this evidence level.
        </p>
      </div>
    );
  }

  const cs = rec.check_size;
  return (
    <div>
      <div className="meta text-[color:var(--muted)]">Cheque</div>
      <div className="mt-1 font-[family-name:var(--font-instrument-serif)] text-[52px] leading-[0.85]">
        {money(rec.amount_usd, rec.currency)}
      </div>
      {cs && (
        <p className="caption mt-1.5 max-w-none text-[color:var(--muted)]">
          Against a {rec.check_size_source ?? "configured"} range of{" "}
          <span className="mono">{money(cs.min, cs.currency)}</span>–
          <span className="mono">{money(cs.max, cs.currency)}</span>, target{" "}
          <span className="mono">{money(cs.target, cs.currency)}</span>.
        </p>
      )}
    </div>
  );
}

export default function DecisionPanel({
  state,
  gate,
  onFocusAxis,
}: {
  state: MemoDissentState;
  /** The gate off the company record — shown even while the memo is still loading. */
  gate: import("@/lib/types").GateOutcome;
  /** Jumps to the axis the recommendation says is governing. */
  onFocusAxis: (a: AxisKey) => void;
}) {
  const { memo, memoMissing, locked, loading, error, openDissent } = state;

  if (memoMissing) {
    return (
      <section
        id="decision"
        className="scroll-mt-32 border border-dashed border-[color:var(--rule)] px-5 py-6"
      >
        <h2 className="meta text-[color:var(--figure)]">Recommendation</h2>
        <p className="mt-2 max-w-[70ch] text-[15px] leading-[1.55] text-[color:var(--muted)]">
          No memo has been written for this company, so there is no computed
          recommendation to show. The gate below still applies — it is computed from the
          axes and does not depend on the memo.
        </p>
        <div className="mt-3">
          <GateBadge gate={gate} />
        </div>
      </section>
    );
  }

  if (!memo) {
    return (
      <section id="decision" className="scroll-mt-32">
        <h2 className="meta mb-3 text-[color:var(--figure)]">Recommendation</h2>
        <Loading label="recommendation" />
      </section>
    );
  }

  const m = memo.data;
  const rec = m.recommendation_detail ?? null;

  return (
    <section
      id="decision"
      className="scroll-mt-32 border-2 bg-[color:var(--ground)]"
      style={{ borderColor: locked ? "var(--rule)" : "var(--accent)" }}
    >
      <header className="flex flex-wrap items-start justify-between gap-4 border-b border-[color:var(--rule)] px-5 py-3">
        <div>
          <h2 className="meta text-[color:var(--figure)]">
            Recommendation — and what is blocking it
          </h2>
          <p className="caption mt-1 max-w-[64ch] text-[color:var(--muted)]">
            The decision, the cheque, and the single component that caps the confidence.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <GateBadge gate={gate} />
          <SourceChip source={memo.source} note={memo.note} />
        </div>
      </header>

      <div className="p-5">
        {error && <ErrorNote message={`Dissent load failed: ${error}`} />}

        {locked ? (
          /* ------------------------------------------------- the server-side lock */
          <div
            className="border-2 border-dashed px-5 py-8 text-center"
            style={{ borderColor: "var(--figure)", background: "var(--ink-09)" }}
          >
            <div className="text-[28px] leading-none" aria-hidden>
              🔒
            </div>
            <p className="mt-2 text-[19px] font-medium text-[color:var(--figure)]">
              Recommendation locked
            </p>
            <p className="mx-auto mt-2 max-w-[52ch] text-[15px] leading-[1.55] text-[color:var(--muted)]">
              The server is withholding it:{" "}
              <em>“{m.recommendation_locked_reason ?? "open the dissent view first"}”</em>.
              You cannot read the recommendation before you have read the case against it.
            </p>
            <button
              type="button"
              onClick={openDissent}
              disabled={loading}
              title="Opens the dissent and asks the server to release the recommendation"
              className="mt-5 border px-5 py-2.5 text-[15px] font-medium tracking-wide text-[color:var(--figure)] transition disabled:opacity-60"
              style={{
                borderColor: "var(--accent)",
                background: "color-mix(in oklab, var(--accent) 14%, transparent)",
              }}
            >
              {loading ? "Opening…" : "Open the dissent to unlock →"}
            </button>
            {loading && (
              <Busy
                className="mx-auto mt-3 max-w-sm"
                budgetMs={TIMEOUT.llm}
                label="Reading the dissent, then re-requesting the memo unlocked"
              />
            )}
          </div>
        ) : (
          <div className="space-y-5">
            {/* The decision sentence, released by the server. */}
            <p
              className="border-l-4 px-4 py-3 text-[17px] leading-[1.5] font-medium text-[color:var(--figure)]"
              style={{
                borderColor: "var(--accent)",
                background: "color-mix(in oklab, var(--accent) 9%, transparent)",
              }}
            >
              {m.recommendation}
            </p>

            {rec ? (
              <>
                <div className="grid gap-5 sm:grid-cols-2">
                  <Cheque rec={rec} />

                  {/* Governing axis — a SELECTION, not an average. */}
                  <div>
                    <div className="meta text-[color:var(--muted)]">Governing axis</div>
                    {rec.governing_axis ? (
                      <>
                        <button
                          type="button"
                          onClick={() =>
                            onFocusAxis(rec.governing_axis!.name as AxisKey)
                          }
                          className="mt-1 flex items-baseline gap-3 text-left"
                        >
                          <span className="font-[family-name:var(--font-instrument-serif)] text-[52px] leading-[0.85] text-[color:var(--accent)] underline decoration-dotted underline-offset-8">
                            {rec.governing_axis.score.toFixed(0)}
                          </span>
                          <span className="meta text-[color:var(--accent)]">
                            {AXIS_LABEL[rec.governing_axis.name as AxisKey] ??
                              rec.governing_axis.name}
                          </span>
                        </button>
                        <p className="caption mt-1.5 max-w-none text-[color:var(--muted)]">
                          The weakest of the three axes governs the cheque. This is the
                          minimum, not a mean — no blended score is computed anywhere.
                        </p>
                      </>
                    ) : (
                      <p className="mt-1.5 text-[15px] leading-[1.55] text-[color:var(--muted)]">
                        No governing axis was reported for this decision.
                      </p>
                    )}
                  </div>
                </div>

                {rec.confidence && <BindingConstraint conf={rec.confidence} />}

                {/* The gate's own reasoning, behind a disclosure — it is long and it is
                    the working rather than the finding. */}
                <details>
                  <summary className="meta cursor-pointer text-[color:var(--accent)]">
                    Why this decision — the gate&apos;s stated reasoning
                  </summary>
                  <p className="mt-2 max-w-[80ch] text-[14px] leading-[1.6] text-[color:var(--muted)]">
                    {rec.reason}
                  </p>
                </details>
              </>
            ) : (
              <p className="caption max-w-none text-[color:var(--muted)]">
                This record carries a written recommendation but no computed cheque or
                confidence breakdown. Nothing is being substituted for the missing half.
              </p>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
