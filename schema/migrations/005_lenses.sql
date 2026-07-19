-- User-authored council lenses (DIFFERENTIATOR §3). Owner: personalisation layer.
-- Applied by scripts/migrate.py; idempotent, so it is safe to re-run.
--
-- WHY A TABLE AND NOT A COLUMN ON vc_profiles. A council is a variable-length set the
-- VC creates, edits and deletes one member at a time, and each member needs its own
-- identity so an edit can address it. A `authored_lenses jsonb` column would make every
-- edit a read-modify-write of the whole council and would give the UI nothing stable to
-- key a row on.
--
-- WHY THIS IS NOT AN EVENT. Same reason as 002: `events` is append-only and every read
-- of it is as_of-scoped on "when the world produced it". A council agent is mutable user
-- state, not an observation about a founder.
--
-- THE LOAD-BEARING CONSTRAINT IS `origin`. A derived lens is an inference the system
-- made from the survey or the decision history and must name the profile field behind
-- it. An authored lens has no such field — the justification is that the VC typed it.
-- The CHECK below makes 'derived' unstorable here, so a lens the user wrote can never
-- present itself as one the system read out of their answers. That separability is what
-- keeps the stated-vs-revealed gap (§2.3) computable.
--
-- NOTHING IS EVER SEEDED INTO THIS TABLE. There is no default council and no starter
-- row. `origin = 'template'` records that the VC knowingly accepted a template and is
-- still a deliberate act by the user; a row that appeared without them asking would be
-- the system inventing a preference on their behalf.

create table if not exists vc_authored_lenses (
    lens_id    uuid primary key default uuid_generate_v4(),
    profile_id uuid not null references vc_profiles(profile_id) on delete cascade,

    -- What the VC calls this agent. Unique per profile so a council cannot contain two
    -- agents the user cannot tell apart in the ranking explanation.
    name       text not null,

    -- The quality this agent adds score for. Read against the company's filtered
    -- evidence graph at scoring time — this is what makes an authored lens move the
    -- ranking rather than decorate it.
    quality    text not null,

    -- The plain-language argument the agent makes. Used verbatim as the persona system
    -- prompt when narration is on.
    persona    text not null,

    -- The VC's own weight. NOT the weight the lens carries in the council: weights are
    -- normalised at compose time so that authored agents share the authored group's
    -- budget and cannot drown the derived ones. See
    -- intelligence/custom_council.py::compose_council.
    weight     double precision not null check (weight >= 0.01 and weight <= 1.0),

    origin     text not null check (origin in ('authored', 'template')),

    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),

    constraint vc_authored_lenses_name_not_blank check (btrim(name) <> ''),
    constraint vc_authored_lenses_quality_not_blank check (btrim(quality) <> ''),
    constraint vc_authored_lenses_persona_not_blank check (btrim(persona) <> '')
);

create unique index if not exists idx_vc_authored_lenses_profile_name
    on vc_authored_lenses (profile_id, lower(btrim(name)));

create index if not exists idx_vc_authored_lenses_profile
    on vc_authored_lenses (profile_id, created_at);

alter table vc_authored_lenses enable row level security;

-- Same posture as 001/002: the API reaches Postgres with the service credential, and
-- the anon/authenticated roles have no business reading another fund's council.
do $$
declare
    role_name text;
begin
    foreach role_name in array array['anon', 'authenticated'] loop
        if exists (select 1 from pg_roles where rolname = role_name) then
            execute format('revoke all on vc_authored_lenses from %I', role_name);
        end if;
    end loop;
end $$;
