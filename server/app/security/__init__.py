"""Abuse controls for the API gateway (Module 3, FE-3).

Two independent concerns, deliberately kept apart:

    ratelimit.py   how *often* a client may call an endpoint
    media.py       whether a submitted file is actually the media it claims

They protect against different things. Rate limiting bounds how much work one
caller can queue; content sniffing stops a caller wasting that work on
something that was never decodable in the first place. Neither substitutes for
the other — an attacker respecting the rate limit can still submit rubbish, and
a well-formed video submitted a thousand times a minute is still an outage.
"""
