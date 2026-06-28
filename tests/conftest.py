"""Shared pytest setup.

Pin Spark's driver IP to loopback before any SparkSession is created, so tests
don't crash on an unresolvable container hostname (JAVA_GATEWAY_EXITED). No-op
for non-Spark tests.
"""

import os

os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
