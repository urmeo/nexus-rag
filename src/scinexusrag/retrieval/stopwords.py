"""Shared stopwords."""

STOP_WORDS: frozenset[str] = frozenset(
    """
    a about above after again against all although an and any are as at be because been before
    being below between both but by can could did do does down during each else every few for
    from further had has have he here how i if in into is it its may me might more most must my
    no not of on once only or other own same shall she should so some such than that the their
    them then there these they this those though through to too under until up very was we were
    what when where which while who whom whose why will with would you your
    """.split()
)
