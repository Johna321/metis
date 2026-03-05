#!/usr/bin/env just --justfile

# display help
@help:
    just --list

# install all dependencies
@install:
    echo "Installing backend dependencies..."
    cd backend && uv sync
    echo "Installing frontend dependencies..."
    cd metis-frontend && bun install

# run backend tests
@test-backend:
    cd backend && uv run pytest

# run frontend tests (we don't have any)
# @test-frontend:
#     cd metis-frontend && bun run test

# run all tests
@test: test-backend #test-frontend

# start development server
@run-dev:
    echo "Starting development server..."
    cd metis-frontend &&  bun tauri dev 

# build and run release
@run-release: build
    echo "Running release build..."
    cd metis-frontend && ./src-tauri/target/release/metis-frontend

# build backend
@build-backend:
    cd backend && uv run metis build

# build frontend
@build-frontend:
    cd metis-frontend &&  bun tauri build

# build both
@build: build-backend build-frontend

# sync IPC types
@sync-types:
    ./scripts/sync-types.sh
