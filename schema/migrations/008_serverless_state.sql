-- Server-side state that USED TO LIVE IN PROCESS MEMORY. Owner: D.
-- Applied by scripts/migrate.py; idempotent, so it is safe to re-run.
--
-- WHY THIS EXISTS. Migration 002 already made this argument for the login throttle:
-- "in the database rather than in process memory because a serverless deployment runs
-- many short-lived processes, and an in-memory counter there is a rate limit that resets
-- on every cold start". Two more pieces of state had the same shape and had not been
-- moved, and on Vercel each request may land on a different lambda, so both broke.
--
-- 1. THE DISSENT LOCK (`dissent_unlocks`). A `set[str]` in api/routers/companies.py.
--    It failed in BOTH directions once there was more than one process:
--      - View the dissent on lambda A, request the memo on lambda B: B never saw it, so
--        the recommendation stayed locked forever and the signature feature looked broken.
--      - On a warm lambda, user A unlocking company X unlocked X FOR EVERY OTHER VISITOR,
--        who then read a recommendation without ever being shown the bear case.
--    The second is the serious one, and it is why this table is keyed by SCOPE and not
--    by company alone. The lock exists so that nobody reads the cheque figure without
--    reading the case against it; a lock that one person can open on another's behalf is
--    not a lock. Scope is per signed-in user, or per anonymous browser — never global.
--
-- 2. PROOF CHALLENGES (`proof_challenges`). api/attest.py recorded issue time and the
--    company a challenge was written for, so a submission could be anchored to a server
--    observation rather than to the founder's own claim about when they started. Issued
--    on one lambda and graded on another, that record was simply absent — and an absent
--    anchor means the elapsed time falls back to SELF-REPORTED, which is precisely the
--    substitution api/attest.py exists to prevent.
--
-- WHY NOT EVENTS. Same reasoning as 002/003/004. `events` is append-only and every read
-- of it is as_of-scoped on `observed_at` — "when the world produced it". "This browser
-- has been shown this bear case" is a fact about a UI session, not an observation about
-- a company, and stamping it with an observed_at would corrupt as_of queries.

create table if not exists dissent_unlocks (
    -- "user:<uuid>" for a signed-in VC, "viewer:<sha256>" for an anonymous browser.
    -- Opaque and prefixed on purpose: the two namespaces can never collide, and a row
    -- here is not a credential — the anonymous form stores the HASH of the viewer
    -- cookie, so a read of this table does not hand the reader a live session.
    scope       text not null,
    -- The company as the ROUTE addressed it (slug or uuid), matching the value the memo
    -- route checks. Deliberately not a foreign key: reseeding rebuilds company rows and
    -- must not silently re-lock a recommendation someone is mid-way through reading.
    company_id  text not null,
    served_at   timestamptz not null default now(),
    primary key (scope, company_id)
);

create index if not exists idx_dissent_unlocks_scope on dissent_unlocks (scope);

create table if not exists proof_challenges (
    challenge_id text primary key,
    -- Nullable: a challenge can be issued before a company is resolved, and a missing
    -- company is reported as "cannot tell" rather than as a mismatch.
    company_id   text,
    issued_at    timestamptz not null default now()
);

alter table dissent_unlocks enable row level security;
alter table proof_challenges enable row level security;

-- Same posture as 001/002/003/004: the API reaches Postgres with the service credential.
-- The anon/authenticated roles have no business reading, and emphatically no business
-- WRITING, a table whose rows unlock recommendations.
do $$
declare
    role_name text;
    tbl text;
begin
    foreach tbl in array array['dissent_unlocks', 'proof_challenges'] loop
        foreach role_name in array array['anon', 'authenticated'] loop
            if exists (select 1 from pg_roles where rolname = role_name) then
                execute format('revoke all on %I from %I', tbl, role_name);
            end if;
        end loop;
    end loop;
end $$;
