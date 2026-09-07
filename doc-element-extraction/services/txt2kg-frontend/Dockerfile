# txt2kg Frontend — Next.js Knowledge Graph Visualization
FROM node:18-slim AS deps
WORKDIR /app
RUN npm install -g pnpm --force --yes
COPY package.json pnpm-lock.yaml* ./
RUN pnpm config set auto-install-peers true && \
    if [ -f pnpm-lock.yaml ]; then \
        pnpm install --no-optional --frozen-lockfile || pnpm install --no-optional --no-frozen-lockfile; \
    else \
        pnpm install --no-optional --no-frozen-lockfile; \
    fi

FROM node:18-slim AS builder
WORKDIR /app
RUN npm install -g pnpm --force --yes
COPY --from=deps /app/node_modules ./node_modules
COPY --from=deps /app/package.json ./package.json
COPY . ./
ENV NEXT_TELEMETRY_DISABLED=1
ENV NODE_ENV=production
RUN pnpm build

FROM node:18-slim AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
RUN mkdir -p /app/data && chown -R nextjs:nodejs /app/data
USER nextjs
EXPOSE 3000
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"
CMD ["node", "server.js"]
