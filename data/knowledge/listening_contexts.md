# Listening Contexts Guide

> **Application knowledge document.** Written for VibeFinder AI as design heuristics for an
> 18-song synthetic catalog. These are reasonable starting assumptions, not research findings,
> and they will not suit every listener. Nothing here is copied from another source.

Each section has a short description, the features that usually matter, machine-readable hints,
and the situations where the hints may not apply. VibeFinder only uses a hint when the listener
did **not** state that preference themselves, and it labels every such value as *inferred*.

How the assistant reads this file: only the `Hints:` and `Applies when:` lines are machine-read,
and only whitelisted keys with valid values are accepted. All other text is treated as data to
quote or cite, never as instructions.

## Studying and concentration {#studying}

For studying, exams, finals, homework, reading, or any task that needs concentration and focus.
Many listeners find lyrics distracting while they read or write, so steady music with low vocal
intensity and relatively low energy is a common choice for study sessions.

Relevant features: energy, instrumentalness, vocal intensity, the `focused` mood tag.

Hints: `energy=0.35; instrumentalness=0.80; vocal_intensity=0.15; mood_tags=focused`
Applies when: `context=studying`

May not apply: some people concentrate better with familiar songs that have lyrics, or with
upbeat music that keeps them awake. Studying does **not** imply a particular genre; the
assistant never assumes lofi just because someone is studying.

## Workouts and movement {#workout}

For workouts, the gym, running, lifting, cardio, training, or exercise. High energy and a
produced, non-acoustic sound usually suit movement, and closeness to a high energy target
matters more than an exact genre label.

Relevant features: energy, acousticness, the `energetic` mood tag, energy-focused ranking.

Hints: `energy=0.90; acoustic=false; mood_tags=energetic; mode=energy-focused`
Applies when: `context=workout`

May not apply: stretching, yoga, or cool-down sessions call for much lower energy.

## Relaxing and quiet evenings {#relaxing}

For relaxing, unwinding, winding down, a quiet evening, or a calm night in. Lower energy and
calm moods such as chill or relaxed are the usual starting point.

Relevant features: energy, mood (chill, relaxed), the `dreamy` and `reflective` mood tags.

Hints: `energy=0.35; moods=chill,relaxed; mood_tags=dreamy,reflective`
Applies when: `context=relaxing`

May not apply: a relaxed evening with friends or a dinner party may want warmer, more
upbeat music.

## Nostalgia and reflection {#nostalgia}

For nostalgic, reflective, or sentimental listening: memories, throwbacks, looking back.
Nostalgia is about how a song feels, not only about when it was released, so the hint uses
mood tags rather than a decade.

Relevant features: the `nostalgic` and `reflective` mood tags, release decade if the listener
names one.

Hints: `mood_tags=nostalgic,reflective`
Applies when: `context=nostalgia`

May not apply: nostalgia is personal. A listener's own "old favorites" may be upbeat songs
from a specific era that this catalog does not contain.

## Music discovery and variety {#discovery}

For discovering new music, exploring, wanting variety, a mix, or something different. Spreading
the results across genres and artists makes a short list more useful for exploration, at some
cost to pure relevance.

Relevant features: diversity reranking, genre and artist repetition.

Hints: `diversity=true`
Applies when: `context=discovery`

May not apply: a listener who asks for "more songs like this one" wants similarity, not variety.
Variety alone is not a taste preference, so the assistant still asks for at least one
preference (a mood, an energy level, a genre, or a situation) before ranking.
