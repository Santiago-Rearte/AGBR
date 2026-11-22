# Blast Radius selector for sqlglot @ eb11aec.
# Heavy layers first (deps, sqlglot checkout, traced test run), light layers last.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1

# 1. The exact environment the commits were labeled in.
COPY requirements.lock /opt/requirements.lock
RUN pip install -q -r /opt/requirements.lock

# 2. sqlglot at the pinned commit, with bytecode precompiled (the runtime disk is read-only).
ARG SQLGLOT_SHA=eb11aec4be8cb2122afe713a4816662c3f0fb2fa
RUN python -c "import io, tarfile, urllib.request; \
data = urllib.request.urlopen('https://github.com/tobymao/sqlglot/archive/${SQLGLOT_SHA}.tar.gz').read(); \
tarfile.open(fileobj=io.BytesIO(data)).extractall('/opt', filter='data')" \
 && mv /opt/sqlglot-${SQLGLOT_SHA} /opt/sqlglot

# 3. Traced run of the whole suite: per-test lines, file reads, instantiated classes.
COPY br/brtrace /opt/br/brtrace
ARG TRACE_N=4
ARG POOL_CPUS=4
RUN cd /opt/sqlglot && mkdir -p /opt/raw \
 && BR_OUT=/opt/raw BR_REPO=/opt/sqlglot SKIP_INTEGRATION=0 PYTHON_CPU_COUNT=${POOL_CPUS} \
    PYTHONPATH=/opt/br/brtrace:/opt/sqlglot \
    timeout 420 python -m pytest tests -p brplugin -p no:cacheprovider -n ${TRACE_N} -q \
      > /opt/raw/pytest.log 2>&1; \
    tail -n 4 /opt/raw/pytest.log; ls /opt/raw | wc -l; \
    BR_REPO=/opt/sqlglot PYTHONPATH=/opt/sqlglot python /opt/br/brtrace/brcore.py /opt/raw/import.trace

# 4. Compact maps from the raw traces.
COPY br/build_maps.py br/statefp.py /opt/br/
RUN cd /opt/sqlglot && PYTHONPATH=/opt/sqlglot python /opt/br/build_maps.py /opt/raw /opt/maps \
 && python -m compileall -q /opt/sqlglot/sqlglot /opt/sqlglot/tests \
 && chmod -R a+rX /opt

# 5. Which tests pass under the runtime runner on the base commit (only those are ever verified).
COPY br/runner.py /opt/br/runner.py
ARG CALIB_N=4
RUN cd /opt/sqlglot && SKIP_INTEGRATION=0 PYTHON_CPU_COUNT=2 \
    python /opt/br/runner.py --calibrate /opt/sqlglot /opt/maps/maps.pkl /opt/maps/verifiable.json ${CALIB_N} \
    2>&1 | tail -n 25 && chmod -R a+rX /opt/maps

# 6. Runtime selector (light layer).
COPY br/selector.py br/analysis.py br/patchlib.py /opt/br/
RUN python -m compileall -q /opt/br && chmod -R a+rX /opt/br
ENV SKIP_INTEGRATION=0 BR_PROBE_VERIFY=0 BR_IMPORT_ALL=0 BR_MIN_LINE_RATE=0 BR_UNIQUE_MAX=0.05 BR_PROBE_DROP=1 BR_PER_HEAVY=0 BR_PER_HEAVY_ALL=0 BR_HEAVY_SEC=1.0 BR_MIN_PROBES=2 BR_DIVERSE_PROBES=4 BR_PROBE_DROP_N=6 BR_PROBES_PER_HEAVY=3 BR_MIN_CHEAP_COVER=1
CMD ["python", "/opt/br/selector.py"]
