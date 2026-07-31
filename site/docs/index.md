---
id: index
locale: en
page: documentation
title: Documentation
description: Start with EvoPolicyGym, understand its evaluation model, and author independent Benchmark distributions.
lead: "Research software for evaluating how a Coding Agent turns bounded Environment feedback into an executable Policy system."
index: D0
docsVersion: v0.3
status: current
slug: /
sidebar_position: 1
---

The documentation follows the current `v0.3` implementation. It describes the
public Python SDK, the bounded Policy ABI, Evaluation and Run semantics, process
execution limitations, and the public Benchmark authoring surface.

## Start here

- [Getting started](./getting-started.md) installs the Kernel and evaluates the
  packaged CartPole baseline.
- [Core concepts](./concepts.md) explains Programs, Policies, Submissions,
  Feedback, Validation, and Assessment.

## Core reference

- [Policy ABI](./policy.md) defines observations, actions, same-Episode state,
  and failure behavior.
- [Evaluation and Runs](./evaluation.md) defines the public search, selection,
  and held-out measurement lifecycle.
- [Runtime and safety](./runtime.md) documents process execution and its
  limitations.

## Extend EvoPolicyGym

- [Benchmark authoring](./authoring.md) describes the public conformance surface
  for independently installable Environment distributions.

The [Environment catalog](/environments/) records the current Benchmark
surface. Historical experiment scores and reruns remain in
[Results](/results/) and are labelled separately from the active runtime.
