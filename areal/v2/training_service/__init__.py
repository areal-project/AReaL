# SPDX-License-Identifier: Apache-2.0

"""Serverless training service — microservice-based training gateway.

Workers are individual SPMD processes (one per 5D-parallel rank), each
wrapped in a synchronous HTTP server.  A data proxy orchestrates a full
5D-parallel group and provides partitioned dispatch.  A router maintains
API key → data proxy mappings.  A gateway provides the public entry
point with authentication and forwarding.
"""
