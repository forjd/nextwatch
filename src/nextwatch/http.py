"""Shared HTTP settings."""

import ssl

# The system CA store, not certifi's bundle: files.grouplens.org fails verification with
# certifi but passes with the OS store, which is also what curl uses.
SSL_CONTEXT = ssl.create_default_context()
