# The plain-English guide to this project

No jargon here. This is the whole study told in everyday words, so anyone can follow what we are doing and why. If a term in the code or the README ever confuses you, come back to this file.

## The one line

We are building a skin-cancer screening app and asking one honest question: **when the photo is bad, does the app still work, and does it know when it is unsure instead of confidently guessing wrong?**

That is the soul of the project. Everything below is just detail.

## The car story (the whole study in one picture)

Think of a car crash-test.

Car makers do not just check that a car drives fine on a smooth, empty road. Anyone can build a car that does that. What they really want to know is what happens when things go wrong: in a crash, does the car protect the person inside?

Our study works the same way. Lots of people build skin-cancer apps that score well on clean, perfect hospital photos, which is the smooth road. Almost nobody checks what happens on the blurry, dark, tilted photos a real person takes on their phone, which is the crash. So we build the apps, confirm they run fine on the smooth road, and then we crash-test them: we damage the photos on purpose, harder and harder, and watch which app holds up.

That is the experiment. The rest is just how we do it.

## Why this matters

In a hospital, photos are perfect. Good camera, good light, the spot centred in frame. In real life, especially somewhere like Nepal where a dermatologist may be far away, a phone photo is often the only screening a person will ever get, and those photos are messy. An app that only works on clean hospital photos is not much use to the people who need it most. So "does it survive bad photos" is not a small technical detail. It is the point.

## The two apps we are comparing

There are two very different ways to build the app, and we built both.

**The Memorizer.** Like a student who crammed thousands of flashcards. Show it a photo and it blurts out one answer with a confidence number: "melanoma, 92%." It is fast and accurate on clean photos. The trouble is it cannot tell you why, and when it is wrong it often still sounds 92% sure.

**The Lookup.** Like a doctor with a big album of past patients. A new photo comes in, the app finds the most similar past cases and lets them vote: "the closest cases were mostly melanoma, so probably melanoma, and here are the cases I based that on." It is slower, and on clean photos it scored lower, but it can show its reasoning. And if a photo looks like nothing it has seen before, that in itself is a useful warning sign.

## What a "fingerprint" means

To find similar past cases, the Lookup first turns each photo into a fingerprint: a long list of numbers that captures how the photo looks. Photos that look alike get similar numbers. Then finding "similar cases" is just finding the closest numbers. This is the same trick from the banknote project, so it is familiar ground.

## The honesty part (the real prize)

Here is the idea that makes this more than just another skin app.

Picture a friend who can never admit they do not know something. You ask for directions, they confidently say "turn left," and you get lost. Now picture a second friend who says "honestly, I'm not sure, better check." For anything that matters, you trust the second friend.

A screening app is the same. One that says "you're fine, 92% sure" while being wrong is dangerous, because it sends a sick person home. One that says "this photo is unclear, go see a doctor" is safe. So we do not only measure how often each app is right. We also measure whether its confidence is honest, especially once the photos get bad. The app that stays honest under pressure is the one worth trusting.

## Where we are right now

We have built and tested both apps on clean photos. The scoreboard so far:

| app | score on clean photos | how often it caught melanoma |
|-----|-----------------------|------------------------------|
| Memorizer | 76% | 56% |
| Lookup | 61% | 47% |

On easy photos, the Memorizer wins. But one thing is already worth knowing: even on perfect photos, the Memorizer misses almost half of melanomas, and it labelled 35 real melanomas as harmless moles. So the "standard" app is shakier than its 76% makes it sound.

## What is next

The crash test. We take the test photos, damage them (blur, darkness, grain, compression, tilt), turn the damage up step by step, and run both apps again. We draw the breakdown curves to see who falls apart faster. Then we check who stays honest about being unsure. That is where the real answer lives, and it is the reason the project is worth a professor's attention.
