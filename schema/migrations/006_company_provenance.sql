-- Company provenance: sourced evidence vs constructed scenario.
-- Applied by scripts/migrate.py; idempotent, so it is safe to re-run.
--
-- WHY THIS IS A COLUMN AND NOT A DERIVED READ. The corpus mixes companies whose
-- evidence was collected from the outside world with companies whose evidence was
-- AUTHORED for this repository — the archetype demo scenarios, and the backtest's
-- matched synthetic controls. Both kinds are legitimate and both need to exist: a
-- cohort of winners alone proves nothing, and a detector with no control is a claim
-- rather than a test. What is NOT legitimate is a reader — or a UI, or a memo, or an
-- outbound draft — being unable to tell which is which. Recomputing that distinction
-- at each call site is how one call site ends up not doing it.
--
-- WHY NOT NULLABLE. A company whose provenance nobody can state is one nobody should
-- be reading evidence off. A NULL here would be exactly the "we are not sure whether
-- this is real" case passing silently through every consumer that forgot to check.
--
-- WHY THE DEFAULT IS 'sourced'. Every runtime writer — the scanners and inbound
-- intake — is reading the real world, so that is the correct default for anything
-- created outside the seed loader. The backfill below is what handles the rows that
-- already existed, and the seed loader sets the value explicitly from the fixtures.

alter table companies
    add column if not exists provenance text not null default 'sourced';

do $$
begin
    if not exists (
        select 1 from pg_constraint where conname = 'companies_provenance_valid'
    ) then
        alter table companies add constraint companies_provenance_valid
            check (provenance in ('sourced', 'constructed'));
    end if;
end $$;

-- Backfill for rows that predate the column.
--
-- This is deliberately keyed on `archetype`, which is the one fact ALREADY in this
-- table that separates the two populations: the archetype fixtures stamp 1..6 on
-- every company they create, and the backtest cohort is loaded with archetype NULL.
-- The authoritative mapping lives in data/seed/provenance.py and is read out of the
-- fixture files themselves; running scripts/seed.py re-asserts it over this backfill
-- and will correct any row this heuristic gets wrong.
--
-- The backtest's four SYNTHETIC controls and its deprioritized failure are the rows
-- this heuristic gets wrong — they carry archetype NULL but are authored, not
-- collected. They are corrected by the seed loader, which reads each cohort member's
-- own `evidence_provenance` field. That correction is not optional: those five are
-- constructed companies sitting in the middle of the cohort that the fame-vs-fitness
-- gate is computed over.
update companies set provenance = 'constructed'
    where archetype is not null and provenance = 'sourced';

create index if not exists idx_companies_provenance on companies (provenance);
