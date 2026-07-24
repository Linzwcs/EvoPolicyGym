# Security

## Supported execution profile

The current `ProcessExecution` profile is for trusted local code. It starts
Policy and Coding Agent subprocesses with the authority of the current
operating-system user. The explicit acknowledgement required by the API does
not provide isolation and must never be described as a sandbox.

Do not evaluate hostile Programs or run untrusted Coding Agents with this
profile. Docker, virtual-machine isolation, durable recovery, and a formal
production operator are not implemented in the active version.

## Reporting

Please report suspected vulnerabilities privately to the project maintainers
before opening a public issue. Include the affected version, execution profile,
reproduction steps, and potential impact.
