-- Inbound applications: company name + deck. Owner: B. SHARED.md S1 — the inbound half
-- of "inbound + outbound -> activate -> ONE FUNNEL".
-- Applied by scripts/migrate.py; idempotent, so it is safe to re-run.
--
-- WHY THIS IS NOT AN EVENT. Same reasoning as 002 and 003. The DECK CONTENT is events —
-- every claim and every integrity finding lands in `events` with its slide id and the
-- deck's own observed_at, via sourcing/bus.py. What lives here is the SUBMISSION: who
-- uploaded which bytes when, and which company row the resolver attached them to. That
-- is a record about us, not an observation about the world, and stamping it with an
-- observed_at would corrupt every as_of query that reads the log.
--
-- WHAT IS DELIBERATELY NOT HERE: a status column.
--
-- The funnel is received -> ingested -> screened -> gated -> decided, and every one of
-- those is DERIVED at read time by sourcing/intake.py::status() from things other parts
-- of the system actually did — deck events in the log, screening events in the log, what
-- intelligence.gate.evaluate returns, whether a human dispositioned an outbound draft.
-- A stored status is a status that drifts the first time a stage fails silently, and the
-- stage most likely to fail silently is the one nobody is watching. There is no setter
-- for these states anywhere in the codebase; adding this column would create one.
--
-- WHAT IS DELIBERATELY NOT HERE (2): a second company identity. `company_id` is whatever
-- memory/resolver.py's matcher returned. A company found by the outbound scanners that
-- then applies inbound gets the SAME company_id, which is what makes the two halves one
-- funnel. The plan's Type 1 guarded failure is "double-count on in/outbound merge".

create table if not exists applications (
    application_id    uuid primary key default uuid_generate_v4(),
    -- Intentionally NOT a foreign key, matching outbound_drafts in 003: the record of
    -- "this deck was submitted for this company on this date" must survive the company
    -- row being rebuilt, which reseeding does routinely.
    company_id        text not null,
    -- What the applicant typed, kept verbatim and separately from the name of the record
    -- they were converged onto. When those two differ, the difference IS the audit trail
    -- of the merge, and overwriting one with the other destroys it.
    submitted_name    text not null,
    company_name      text not null,
    submitted_by      text,
    founder_name      text,
    founder_email     text,
    -- The entity memory/resolver.py returned for the submitting founder, when one was
    -- given. Null is the normal case: a deck arrives with no contact details.
    founder_entity_id text,
    deck_filename     text not null,
    -- The dedupe key, with company_id. Re-uploading identical bytes is one application,
    -- not two — the other half of the double-count guard.
    deck_sha256       text not null,
    deck_bytes        bigint not null,
    deck_path         text not null,
    -- JSON as text, NOT jsonb — see 002/003. Everything above memory/db.py is written in
    -- the SQLite dialect and translated centrally.
    --
    -- The resolver's decision AT THE MOMENT OF INTAKE: status, score, which existing name
    -- it matched, the near-misses it refused to merge on, and its rationale. Snapshotted
    -- rather than recomputed, because an audit of a merge has to show what the matcher
    -- believed when it merged, not what it would believe now against a different corpus.
    convergence       text not null default '{}',
    received_at       timestamptz not null default now()
);

create index if not exists idx_applications_company on applications (company_id);
create unique index if not exists idx_applications_dedupe
    on applications (company_id, deck_sha256);

alter table applications enable row level security;

-- Same posture as 001/002/003: the API reaches Postgres with the service credential.
-- The anon/authenticated roles have no business reading unreviewed applications.
do $$
declare
    role_name text;
begin
    foreach role_name in array array['anon', 'authenticated'] loop
        if exists (select 1 from pg_roles where rolname = role_name) then
            execute format('revoke all on applications from %I', role_name);
        end if;
    end loop;
end $$;
