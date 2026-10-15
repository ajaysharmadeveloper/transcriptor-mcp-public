"""The contracts that hold the system together.

Which members are ``async def`` and which are plain ``def`` *is* the concurrency
design, encoded in the type signature.  Do not tidy a sync port into an async one.
"""
