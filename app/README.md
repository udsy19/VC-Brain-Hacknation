# app — dashboard (owner: D)

Not scaffolded here on purpose: `create-next-app` writes ~40 files and D should own
that tree from the first commit rather than inherit someone else's choices.

```bash
pnpm create next-app@latest . --ts --tailwind --app --eslint --src-dir=false
pnpm dlx shadcn@latest init
```

Backend runs at `http://localhost:8000` (CORS already allows `localhost:3000`).

Routes to build (see D.md):
- `/`               ranked list + momentum, thesis config, NL query
- `/company/[id]`   three axes side by side, trace drill-down, memo|dissent split view
- `/backtest`       calibration report

Two rules the UI must not break:
- **Never render a single blended score.** Three axes, always separate.
- The recommendation is locked server-side until dissent is opened — don't work around it.
