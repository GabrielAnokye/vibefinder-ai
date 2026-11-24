# Music Features and Recommendation Strategies Guide

> **Application knowledge document.** Written for VibeFinder AI to explain what its song
> features and ranking strategies mean **in this project's synthetic catalog**. It is not an
> authoritative description of how music services work, and nothing here is copied from another
> source. Catalog numbers are deliberately left out: they are computed from `data/songs.csv`
> at run time so they cannot go out of date.

## Genre versus mood {#genre-vs-mood}

Genre describes a song's style (rock, lofi, jazz). Mood describes how it feels (happy, chill,
sad). A rock song can be moody and a pop song can be intense, so the two are scored separately.
Labels match exactly after case and spacing are normalized: "indie pop" does not count as "pop".
When a listener names several genres ("rock or hip-hop"), a song matching any of them earns the
single genre contribution once.

## Energy {#energy}

Energy is a 0 to 1 value for how intense and active a song feels. Words such as calm, quiet,
mellow, or gentle point to low energy; energetic, upbeat, or high energy point to high energy.
The score rewards closeness to the target, not loudness: a 0.40 song is a perfect fit for a 0.40
target.

## Acousticness {#acousticness}

Acousticness is a 0 to 1 estimate of how acoustic (unplugged, unamplified) a song sounds. A
listener who asks for acoustic music earns points from high acousticness; one who prefers
produced or electronic sound earns points from low acousticness.

## Instrumentalness {#instrumentalness}

Instrumentalness is a 0 to 1 estimate of how much of a song has no vocals. High values suit
requests for instrumental music, music without lyrics, or music without distracting vocals.
It is only scored when advanced scoring is on.

## Vocal intensity {#vocal-intensity}

Vocal intensity is a 0 to 1 estimate of how prominent and forceful the vocals are. Low values
suit background or study listening; high values suit sing-along or performance-driven songs.
It is only scored when advanced scoring is on.

## Popularity {#popularity}

Popularity is a synthetic 0 to 100 value. Asking for hits or popular songs points to a high
target; asking for hidden gems or lesser-known songs points to a low target. The values are
invented for fictional songs and do not come from any streaming service.

## Release decade {#release-decade}

Release decade groups songs by era (1990s, 2000s, 2010s, 2020s). A decade earns points only on
an exact match. Feeling nostalgic is not the same as asking for an old song, so the assistant
never infers a decade from the word "nostalgic".

## Mood tags {#mood-tags}

Mood tags add detail beyond the single mood label: nostalgic, euphoric, reflective, aggressive,
dreamy, focused, uplifting, melancholic, energetic. A song earns tag points in proportion to how
many of the requested tags it carries. Tags are scored only when advanced scoring is on.

## Ranking strategies {#strategies}

Baseline ranking weights genre most (genre 4, mood 3, energy 3, acoustic 1). Mood-first ranking
lets mood matter most and is a good fit when a listener describes a feeling but no genre.
Energy-focused ranking doubles the energy weight and suits requests where intensity matters
more than labels, such as workouts. Every strategy uses the same scoring formula; only the
weights change.

## Diversity and relevance {#diversity}

Diversity reranking subtracts a penalty when a genre or artist repeats, which spreads a short
list across more styles. The penalty changes the order only; the relevance score shown for each
song is never altered. A penalty is not a guarantee: when a listener says "don't repeat
artists", that rule is enforced separately as a hard constraint.

## Small catalog limitations {#small-catalog}

The catalog is small and uneven, so some combinations of preferences cannot all be satisfied.
When preferences conflict with what the catalog contains, the top result is the best available
compromise, not a perfect match. The assistant reports which stated preferences the results
miss instead of hiding the gap.

## Genres not in the catalog {#genres-not-in-catalog}

Heavy genres such as metal, punk, and hard rock are not in the catalog. When a listener asks for
one as a soft preference, the closest songs by sound are high-energy, non-acoustic tracks tagged
aggressive or energetic; these are substitutes, not examples of the requested genre. When a
listener says "only" that genre, no substitute is offered.

Hints: `energy=0.90; acoustic=false; mood_tags=aggressive,energetic`
Applies when: `unavailable_genre=metal|punk|hard rock|heavy metal|grunge`
