# A.R.C.H.I.E. - Adaptive Retrieval, Curation, and Hierarchical Information Extraction

ARCHIE provides the web search function for Jarvis by running [SearXNG](https://searxng.org/), a privacy-focused metasearch engine. SearXNG aggregates results from multiple upstream engines behind a single query endpoint used by the Open WebUI service (loki).

## Services

ARCHIE is composed of two Docker Compose services, defined in [../compose.yaml](../compose.yaml):

- **archie-core** - the SearxNG search engine itself (`searxng/searxng` image).
- **archie-data** - a Valkey server used by SearXNG for caching and the limiter.

Configuration is maintained locally under [`etc/searxng/`](../archie/etc/searxng/) (settings, limiter) and mounted into the `archie-core` container. Related runtime values are set via the environment variables beginning with `SEARXNG` in [../.env](../.env):

| Variable              | Purpose                                |
| --------------------- | -------------------------------------- |
| `SEARXNG_VERSION`     | SearXNG image tag                     |
| `SEARXNG_HOST`        | Host address to bind the service on   |
| `SEARXNG_PORT`        | Port the service listens on           |
| `SEARXNG_QUERY_URL`   | Query URL used by Jarvis (loki)       |

The `server.secret_key` value in [`etc/searxng/settings.yml`](../archie/etc/searxng/settings.yml) ships as the placeholder `change_me`. Replace it with a long, random string before use; SearXNG uses it to sign session cookies and limiter state. To generate one:

```sh
openssl rand -hex 32
```

Then paste the output into settings as `secret_key: "<generated-value>"` (keep at least 32 characters).

For details on configuration options, refer to the [SearXNG documentation](https://docs.searxng.org/).

## References

- [SearXNG](https://searxng.org/)
- [SearXNG Documentation](https://docs.searxng.org/)
