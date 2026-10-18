// Railway project for nextwatch (docs/plans/09-deploy.md).
//   cd .railway && npm install && railway link && railway config plan
// Secrets are set in the Railway dashboard; preserve() keeps them out of this file.
import { defineRailway, github, image, preserve, project, service, volume } from "railway/iac";

export default defineRailway(() => {
  const pgdata = volume("pgdata", { sizeMB: 5000 });

  // Railway's managed Postgres has no pgvector, so run the same image as compose.yaml.
  const db = service("postgres", {
    source: image("pgvector/pgvector:pg17"),
    volumeMounts: { "/var/lib/postgresql/data": pgdata },
    env: {
      POSTGRES_USER: "nextwatch",
      POSTGRES_DB: "nextwatch",
      POSTGRES_PASSWORD: preserve(),
      // A volume's root holds lost+found, which initdb refuses; use a subdirectory.
      PGDATA: "/var/lib/postgresql/data/pgdata",
    },
  });

  const databaseUrl =
    "postgresql://nextwatch:${{postgres.POSTGRES_PASSWORD}}@${{postgres.RAILWAY_PRIVATE_DOMAIN}}:5432/nextwatch";

  const web = service("web", {
    source: github("forjd/nextwatch", { branch: "main" }),
    start: "nextwatch web --host :: --port 8000",
    preDeploy: "nextwatch migrate",
    healthcheck: "/healthz",
    healthcheckTimeout: 120,
    env: {
      DATABASE_URL: databaseUrl,
      PORT: "8000",
      APP_PASSWORD: preserve(),
      SESSION_SECRET: preserve(),
    },
  });

  const daily = service("daily", {
    source: github("forjd/nextwatch", { branch: "main" }),
    start: "nextwatch daily",
    // 08:30 UTC: after TMDB publishes the daily exports at 08:00.
    deploy: { cronSchedule: "30 8 * * *", restartPolicyType: "NEVER" },
    env: {
      DATABASE_URL: databaseUrl,
      TMDB_API_TOKEN: preserve(),
      OPENROUTER_API_KEY: preserve(),
      BACKUP_GITHUB_TOKEN: preserve(),
      BACKUP_GITHUB_REPO: "forjd/nextwatch-backups",
      VIBE_BUDGET_USD: "5",
    },
  });

  return project("nextwatch", { resources: [db, pgdata, web, daily] });
});
