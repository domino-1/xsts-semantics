#!/usr/bin/env bash

set -euxo pipefail

kxsts --help

kxsts run --verbose examples/imp/sumto10.imp --env 'x=0,y=1' --env b1=false --env b2=true

kxsts prove --verbose examples/imp/specs/imp-sum-spec.k IMP-SUM-SPEC sum-spec

kxsts show --verbose IMP-SUM-SPEC sum-spec
