"""hlbench CLI.

This package lives OUTSIDE ``src/hlbench/`` because it's a consumer
of the server library (CLAUDE.md invariant 9 + lib/consumer separation
in project memory). The CLI talks to a running ``hlbench serve``
process over HTTP.

Subcommands::

    hlbench init      --env <id> --dir <ws>          create a workspace
    hlbench serve     --workspace <ws> [--port 8765] start HTTP server
    hlbench info      [--url http://host:port]       GET /info
    hlbench submit    --env-instances <range> [--url ...]
    hlbench finalize  [--url http://host:port]
"""
