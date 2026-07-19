-- VC accounts, sessions, and the two preference sources. Owner: personalisation layer.
-- Applied by scripts/migrate.py; idempotent, so it is safe to re-run.
--
-- WHY THESE ARE NOT EVENTS. `events` is append-only by DB trigger and every read of it
-- is as_of-scoped on `observed_at` — "when the world produced it". A profile is MUTABLE
-- state: the user edits their survey, re-uploads decisions, changes a red line. Storing
-- that in the event log would mean stamping an observed_at of "when the user filled in a
-- form", which is not a fact about the world and would corrupt every as_of query that
-- reads the log. So: ordinary tables with ordinary update semantics.
--
-- WHY SURVEY AND DECISIONS ARE SEPARATE TABLES. DIFFERENTIATOR §0/§2.3 — the
-- stated-vs-revealed gap is only computable if the two preference sources are never
-- merged. One `profile_data jsonb` blob would be less code and would delete the feature.

create extension if not exists "uuid-ossp";

create table if not exists users (
    user_id       uuid primary key default uuid_generate_v4(),
    email         text not null unique,   -- stored already lowercased by the app
    password_hash text not null,          -- argon2id. NEVER selected into an API response.
    created_at    timestamptz not null default now()
);

create table if not exists sessions (
    session_id uuid primary key default uuid_generate_v4(),
    -- The sha256 of the token, not the token. A read of this table does not hand the
    -- reader a set of live sessions.
    token_hash text not null unique,
    user_id    uuid not null references users(user_id) on delete cascade,
    created_at timestamptz not null default now(),
    expires_at timestamptz not null
);

create index if not exists idx_sessions_user on sessions (user_id);
create index if not exists idx_sessions_expires on sessions (expires_at);

-- Per-email login throttle. In the database rather than in process memory because a
-- serverless deployment runs many short-lived processes, and an in-memory counter there
-- is a rate limit that resets on every cold start.
create table if not exists login_attempts (
    email        text primary key,
    failures     int not null default 0,
    first_failure_at timestamptz not null default now(),
    locked_until timestamptz
);

create table if not exists vc_profiles (
    profile_id       uuid primary key default uuid_generate_v4(),
    user_id          uuid not null unique references users(user_id) on delete cascade,
    fund_name        text,
    -- JSON arrays as text, NOT jsonb. Everything above memory/db.py is written once in
    -- the SQLite dialect and translated centrally, so a column whose write needs a
    -- psycopg-specific Jsonb() wrapper would put a backend branch in the storage layer.
    -- These are small lists that are only ever read whole; jsonb buys nothing here.
    focus_sectors    text not null default '[]',
    stated_red_lines text not null default '[]',
    created_at       timestamptz not null default now(),
    updated_at       timestamptz not null default now()
);

-- STATED preference. One row per answered question; an unanswered question has NO row,
-- which is what lets the derivation count real answers instead of assuming a default.
create table if not exists vc_survey_answers (
    profile_id  uuid not null references vc_profiles(profile_id) on delete cascade,
    question_id text not null,
    choice      text not null check (choice in ('a', 'b')),
    answered_at timestamptz not null default now(),
    primary key (profile_id, question_id)
);

-- REVEALED preference. `source_row` is the 1-based line of the uploaded file and is the
-- provenance handle every derived value points back at.
create table if not exists vc_decisions (
    decision_id uuid primary key default uuid_generate_v4(),
    profile_id  uuid not null references vc_profiles(profile_id) on delete cascade,
    company     text not null,
    sector      text,
    stage       text,
    decision    text not null check (decision in ('invested', 'passed', 'watched')),
    decided_on  date,
    rationale   text,
    outcome     text,
    source_row  int,
    uploaded_at timestamptz not null default now()
);

create index if not exists idx_vc_decisions_profile on vc_decisions (profile_id);

alter table users             enable row level security;
alter table sessions          enable row level security;
alter table login_attempts    enable row level security;
alter table vc_profiles       enable row level security;
alter table vc_survey_answers enable row level security;
alter table vc_decisions      enable row level security;

-- Same posture as 001: the API reaches Postgres with the service credential, and the
-- anon/authenticated roles have no business reading a password hash or a session token.
do $$
declare
    role_name text;
begin
    foreach role_name in array array['anon', 'authenticated'] loop
        if exists (select 1 from pg_roles where rolname = role_name) then
            execute format(
                'revoke all on users, sessions, login_attempts, vc_profiles, '
                'vc_survey_answers, vc_decisions from %I',
                role_name
            );
        end if;
    end loop;
end $$;
