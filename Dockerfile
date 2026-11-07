FROM node:22-bookworm-slim AS frontend
WORKDIR /ui
COPY app/package.json app/package-lock.json ./
RUN npm ci
COPY app/ ./
RUN npm run build

FROM python:3.12-slim-bookworm AS python-deps
WORKDIR /build
COPY api/requirements.txt ./requirements.txt
RUN pip install --prefix=/install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.9.1 \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt

FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 STATEMATCH_DB=/data/statematch.sqlite3
WORKDIR /app
COPY --from=python-deps /install /usr/local
RUN groupadd --gid 10001 statematch && useradd --uid 10001 --gid statematch --no-create-home statematch \
    && mkdir -p /data && chown statematch:statematch /data
COPY api/ ./api/
COPY modeling/ ./modeling/
COPY --from=frontend /ui/dist ./app/dist
USER statematch
EXPOSE 8000
VOLUME ["/data"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)" || exit 1
CMD ["python", "-m", "uvicorn", "app.main:app", "--app-dir", "api", "--host", "0.0.0.0", "--port", "8000"]
