"""A small, real, human-judged retrieval corpus for RAG quality evaluation.

24 documents across 6 distinct domains + 16 golden queries with hand-assigned
relevance. The queries deliberately mix two regimes so the eval measures both:

* **semantic / paraphrase** queries that share almost NO tokens with their target
  document (these reward a real embedder over lexical overlap), and
* **rare-term / exact** queries that hinge on a distinctive token — a cultivar name,
  a ticker, a specific figure (these are where BM25 in the hybrid leg helps).

Relevance is single-document binary (the clearly-correct source for each query),
which is enough for recall@k / MRR / nDCG@k with the standard binary-gain formula.
Keys (not ids) tie queries to docs; the eval maps key -> document_id after ingest.
"""

from __future__ import annotations

# (key, title, text)
DOCS: list[tuple[str, str, str]] = [
    # --- permaculture / farming -------------------------------------------------
    (
        "bees",
        "Apiary basics",
        "Honeybees forage nectar from blossoms and convert it into honey inside the hive; a strong colony also pollinates nearby orchards and raises fruit set.",
    ),
    (
        "foodforest",
        "Food forests",
        "A food forest layers canopy trees, fruiting shrubs, herbs and ground cover to mimic a natural woodland, building soil and yielding food with little tillage.",
    ),
    (
        "ducks",
        "Integrated ducks",
        "Runner ducks patrol vegetable beds eating slugs and snails, and their manure fertilises a pond whose nutrient-rich water is pumped onto the orchard.",
    ),
    (
        "compost",
        "Hot composting",
        "A thermophilic compost pile reaching sixty degrees Celsius kills weed seeds and pathogens within days; turning it aerates the heap and speeds decomposition.",
    ),
    # --- finance / markets ------------------------------------------------------
    (
        "ratehike",
        "Monetary tightening",
        "When prices rise too quickly a central bank lifts its policy interest rate, making credit dearer, cooling demand and pulling inflation back toward target.",
    ),
    (
        "nabil",
        "A bank stock",
        "Shares of NABIL, a large commercial bank, rallied after it reported higher quarterly profit and announced a generous cash dividend for shareholders.",
    ),
    (
        "etf",
        "Index funds",
        "An index fund holds every constituent of a market benchmark in proportion, giving a saver cheap, diversified exposure without picking individual companies.",
    ),
    (
        "shortsell",
        "Short selling",
        "A short seller borrows a security and sells it, hoping to buy it back cheaper later; the trade profits when the price falls but loses without limit if it climbs.",
    ),
    # --- health / nutrition -----------------------------------------------------
    (
        "omega3",
        "Dietary fats",
        "Oily fish such as salmon and mackerel are rich in omega-3 fatty acids, which support heart and brain health and help lower triglyceride levels.",
    ),
    (
        "sleep",
        "Sleep hygiene",
        "Adults who keep a regular bedtime, avoid late caffeine and dim screens before sleeping tend to fall asleep faster and wake feeling more rested.",
    ),
    (
        "hydration",
        "Water needs",
        "During hot or strenuous activity the body loses fluid and electrolytes through sweat, so drinking water steadily prevents the headache and fatigue of dehydration.",
    ),
    (
        "vitd",
        "Vitamin D",
        "Skin makes vitamin D from sunlight, and the nutrient helps the gut absorb calcium; people with little sun exposure often need a dietary supplement.",
    ),
    # --- astronomy / space ------------------------------------------------------
    (
        "neutron",
        "Stellar remnants",
        "When a massive star exhausts its fuel its core collapses in a supernova, leaving behind an extraordinarily dense neutron star only kilometres across.",
    ),
    (
        "blackhole",
        "Event horizons",
        "A black hole's gravity is so strong that past a boundary called the event horizon not even light can escape, so the object itself emits nothing.",
    ),
    (
        "exoplanet",
        "Other worlds",
        "Astronomers detect planets around distant stars by watching a star dim slightly as a world transits across its face, or wobble under the planet's pull.",
    ),
    (
        "comet",
        "Icy visitors",
        "A comet is a body of ice and dust that grows a glowing tail as it nears the Sun and its frozen gases vaporise, streaming away from the solar wind.",
    ),
    # --- computing / AI ---------------------------------------------------------
    (
        "embedding",
        "Vector embeddings",
        "An embedding model maps a sentence to a list of numbers so that texts with similar meaning sit close together, enabling search by meaning rather than keywords.",
    ),
    (
        "rrf",
        "Rank fusion",
        "Reciprocal Rank Fusion combines two ranked result lists by summing one over a constant plus each rank, a simple way to blend keyword and vector retrieval.",
    ),
    (
        "gpu",
        "Accelerators",
        "A graphics processing unit performs thousands of arithmetic operations in parallel, which is why training large neural networks is far faster on a GPU than a CPU.",
    ),
    (
        "cache",
        "Caching",
        "A cache stores the result of an expensive computation or query so that a later identical request is served quickly from memory instead of being recomputed.",
    ),
    # --- geography / Nepal ------------------------------------------------------
    (
        "everest",
        "Highest peak",
        "Mount Everest, on the border of Nepal and Tibet, is the tallest mountain above sea level, drawing climbers who acclimatise for weeks before a summit push.",
    ),
    (
        "bardaghat",
        "A Terai town",
        "Bardaghat lies in Nepal's lowland Terai belt, a warm, fertile plain where farmers grow rice, lentils and vegetables across the monsoon and winter seasons.",
    ),
    (
        "kathmandu",
        "Capital valley",
        "Kathmandu, Nepal's capital, sits in a bowl-shaped valley ringed by hills and is dotted with centuries-old temples and bustling market squares.",
    ),
    (
        "monsoon",
        "Summer rains",
        "Nepal's monsoon arrives in June as moist winds off the Bay of Bengal, dropping most of the year's rain over three months and filling the rivers and paddy fields.",
    ),
    # --- CONFUSABLE cluster: near-identical bank stocks distinguished only by name.
    # Semantic content is nearly the same across all four, so a dense embedder must lean
    # on the proper noun to disambiguate — the regime where the hybrid BM25 leg can help.
    (
        "nica",
        "A commercial bank",
        "Shares of NICA, a large commercial bank, advanced after it posted higher quarterly profit and declared a cash dividend for its shareholders.",
    ),
    (
        "scbl",
        "A foreign-backed bank",
        "Shares of Standard Chartered, a foreign-backed commercial bank, edged up after it reported a steady quarterly profit and proposed a cash dividend to shareholders.",
    ),
    (
        "hbl",
        "A pioneer bank",
        "Shares of Himalayan Bank, a long-established commercial bank, gained after stronger quarterly profit and a proposed cash dividend to its shareholders.",
    ),
    # --- CONFUSABLE cluster: similar stellar end-states distinguished by a key term.
    (
        "whitedwarf",
        "Dying small stars",
        "A white dwarf is the dense cooling core a sun-like star leaves after shedding its outer layers, held against gravity by electron degeneracy pressure.",
    ),
    (
        "redgiant",
        "Swollen old stars",
        "As a sun-like star exhausts its core hydrogen it swells into a red giant, its bloated outer layers cooling to a reddish glow before it sheds them.",
    ),
]

# (query, [relevant_keys], label) — single clearly-correct source per query.
# label encodes the regime so the scorecard can be read by query type.
CASES: list[tuple[str, list[str], str]] = [
    # semantic / paraphrase (little-to-no token overlap with the target doc)
    (
        "how do pollinating insects turn flower nectar into a sweet syrup",
        ["bees"],
        "semantic",
    ),
    (
        "a way a bank fights rising prices by making borrowing more expensive",
        ["ratehike"],
        "semantic",
    ),
    (
        "multi-tiered regenerative planting that imitates a natural wood",
        ["foodforest"],
        "semantic",
    ),
    ("which foods help your heart with healthy fish oils", ["omega3"], "semantic"),
    (
        "the ultra-dense leftover core after a giant star explodes",
        ["neutron"],
        "semantic",
    ),
    (
        "searching documents by meaning instead of matching exact words",
        ["embedding"],
        "semantic",
    ),
    (
        "the world's loftiest summit straddling the Nepal-Tibet frontier",
        ["everest"],
        "semantic",
    ),
    (
        "borrowing an asset to sell it now and repurchase it cheaper later",
        ["shortsell"],
        "semantic",
    ),
    (
        "hardware that runs many calculations at once to speed up model training",
        ["gpu"],
        "semantic",
    ),
    (
        "staying well-watered so you avoid headaches during a hot workout",
        ["hydration"],
        "semantic",
    ),
    # rare-term / exact (a distinctive token favouring the lexical/hybrid leg)
    ("NABIL quarterly profit and dividend", ["nabil"], "lexical"),
    ("Reciprocal Rank Fusion", ["rrf"], "lexical"),
    ("Bardaghat Terai farmers rice lentils", ["bardaghat"], "lexical"),
    ("vitamin D calcium absorption supplement", ["vitd"], "lexical"),
    ("event horizon black hole light escape", ["blackhole"], "lexical"),
    ("monsoon Bay of Bengal Nepal June rain", ["monsoon"], "lexical"),
    # disambiguation among near-identical docs — the exact proper noun / term is the
    # only reliable signal, so this is where dense can slip and hybrid (BM25) can help.
    ("NICA Bank quarterly dividend", ["nica"], "disambig"),
    ("Standard Chartered bank profit dividend", ["scbl"], "disambig"),
    ("Himalayan Bank shareholders quarterly profit", ["hbl"], "disambig"),
    ("white dwarf electron degeneracy pressure", ["whitedwarf"], "disambig"),
    ("red giant swollen outer layers reddish", ["redgiant"], "disambig"),
]
