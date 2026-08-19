"""TTS test data: sentences, edge cases, text length buckets, long-form passages."""
from __future__ import annotations

from pathlib import Path

_HARVARD_PATH = Path(__file__).parent / "harvard_sentences.txt"


def _load_harvard() -> list[str]:
    if _HARVARD_PATH.exists():
        lines = _HARVARD_PATH.read_text().splitlines()
        return [l.strip() for l in lines if l.strip()]
    return []


def get_naturalness_sentences() -> list[str]:
    """200 sentences for Test 2.1 naturalness evaluation."""
    harvard = _load_harvard()
    extra = [
        "The neural network achieved state-of-the-art performance on the benchmark.",
        "Please confirm your appointment for tomorrow at three o'clock.",
        "The quarterly earnings report exceeded analyst expectations by twelve percent.",
        "We regret to inform you that your flight has been delayed by two hours.",
        "The temperature today will reach a high of seventy-two degrees Fahrenheit.",
        "Your order has been shipped and will arrive within three to five business days.",
        "The new software update includes several important security patches.",
        "Press one for English, press two for Spanish, or stay on the line.",
        "The meeting has been rescheduled to next Tuesday at two thirty PM.",
        "Thank you for calling customer support. How can I assist you today?",
    ]
    sentences = harvard[:190] + extra
    return sentences[:200] if len(sentences) >= 200 else sentences


def get_intelligibility_sentences() -> dict[str, list[str]]:
    """Sentences by category for Test 2.2 round-trip WER."""
    return {
        "harvard": _load_harvard()[:80],
        "technical": [
            "The CPU utilization spiked to ninety-eight percent during peak load.",
            "Please authenticate using your two-factor authentication token.",
            "The API endpoint returns a JSON response with a status code of two hundred.",
            "Configure the subnet mask to two five five point two five five point two five five point zero.",
            "The database query took three hundred milliseconds to execute.",
            "Deploy the containerized application using the Kubernetes orchestration platform.",
            "The machine learning model achieved an F1 score of zero point nine three.",
            "Enable HTTPS by installing the SSL certificate on port four four three.",
            "The microservices architecture uses REST and gRPC for inter-service communication.",
            "Allocate four gigabytes of RAM and eight CPU cores to the virtual machine.",
        ] * 5,
        "numbers_basic": [
            # Normaliser-friendly: standard currency / units / plain integers.
            # Expected post-normalisation: near-zero WER on any TTS that
            # pronounces the digits correctly.
            "The price is one thousand two hundred and thirty-four dollars and fifty-six cents.",
            "The patient's temperature is ninety-eight point six degrees Fahrenheit.",
            "The population of the city is approximately two point three million people.",
            "Multiply three hundred and forty-two by seventeen to get five thousand eight hundred and fourteen.",
        ] * 5,
        "numbers_edge": [
            # Normaliser-limited surface forms: phone numbers, coords,
            # version numbers, dates, alphanumeric IDs, clock format. These
            # are intentionally hard for jiwer-comparison even after the
            # Whisper EnglishTextNormalizer; surviving WER here is measurement
            # noise, not TTS intelligibility loss.
            "Call us at one eight hundred five five five zero one two three.",
            "The coordinates are forty point seven one degrees north, seventy-four point zero one degrees west.",
            "Version three point fourteen one five nine two of the software is now available.",
            "The meeting is scheduled for the fifteenth of March twenty twenty-five.",
            "Your confirmation number is A B C one two three four five six seven.",
            "The flight departs at six forty-five AM and arrives at ten fifteen PM.",
        ] * 5,
        "conversational": [
            "Hey, can you remind me to call Sarah at five o'clock this afternoon?",
            "I'd like to book a table for four people this Saturday evening.",
            "Could you please repeat that? I didn't quite catch what you said.",
            "The weather's been really nice lately, hasn't it?",
            "I'm running a bit late — could you tell them I'll be there in ten minutes?",
            "Do you have any recommendations for good Italian restaurants nearby?",
            "Sorry to bother you, but I think there might be a mistake on my bill.",
            "That's absolutely fantastic news! Congratulations on your promotion.",
            "I've been trying to reach you all morning. Is everything okay?",
            "Let's catch up over coffee sometime next week if you're free.",
        ] * 5,
    }


def get_text_length_buckets() -> dict[str, list[str]]:
    """Text samples grouped by length for Test 2.5 latency bucketing."""
    return {
        "short_10_words": [
            "The sun sets over the distant mountains.",
            "Please leave a message after the tone.",
            "Your call is very important to us.",
            "The package will arrive on Thursday.",
            "Thank you for your patience today.",
            "Press star to return to the main menu.",
            "The office is closed on weekends.",
            "Please enter your four-digit PIN now.",
            "Your balance is currently five hundred dollars.",
            "The next train departs in seven minutes.",
        ],
        "medium_25_words": [
            "The artificial intelligence revolution is transforming industries across the globe, from healthcare and finance to transportation and entertainment sectors.",
            "Thank you for calling the customer service department. All of our representatives are currently busy, but your call is important to us.",
            "The quarterly financial results showed a significant improvement in revenue growth, driven primarily by strong performance in the European and Asian markets.",
            "Please be advised that the scheduled maintenance window will take place this Saturday between midnight and four o'clock in the morning Pacific time.",
            "The research team has successfully developed a new machine learning algorithm that improves prediction accuracy by twenty-three percent over the previous model.",
        ],
        "long_75_words": [
            (
                "Welcome to the annual technology conference. This year's event brings together industry leaders, researchers, and innovators from around the world to discuss "
                "the latest advancements in artificial intelligence, cloud computing, and quantum information science. Over the next three days, you will have the opportunity "
                "to attend keynote presentations, participate in hands-on workshops, and network with over five thousand attendees from more than sixty different countries."
            ),
            (
                "The Board of Directors is pleased to announce the appointment of Dr. Alexandra Chen as the new Chief Executive Officer, effective the first of next month. "
                "Dr. Chen brings over twenty years of experience in the technology industry, having previously served as President of Global Operations at several Fortune 500 companies. "
                "Under her leadership, the company expects to accelerate its digital transformation strategy and expand into new international markets."
            ),
        ],
        "very_long_150_words": [
            (
                "Good morning and welcome to today's earnings call. I'm joined by our Chief Financial Officer and Chief Operating Officer to discuss our third-quarter results. "
                "Before we begin, I'd like to remind everyone that certain statements made during this call may constitute forward-looking statements within the meaning of the "
                "Securities Exchange Act. These statements involve risks and uncertainties that could cause actual results to differ materially from those expressed or implied. "
                "For the third quarter, we reported total revenue of two point four billion dollars, representing a year-over-year increase of eighteen percent. "
                "Gross margin expanded by one hundred and fifty basis points to forty-two percent, driven by operational efficiencies and favorable product mix. "
                "Operating income increased to three hundred and twenty million dollars, and diluted earnings per share came in at one dollar and forty-seven cents, "
                "exceeding consensus estimates by twelve cents. We are raising our full-year guidance range to reflect the strong momentum we are seeing across all business segments."
            ),
        ],
        "extra_long_300_words": [
            (
                "Ladies and gentlemen, distinguished guests, and valued colleagues, it is my distinct honor and privilege to welcome you to this extraordinary gathering of minds "
                "that has brought together some of the most brilliant innovators, entrepreneurs, and thought leaders from every corner of our interconnected world. "
                "We stand today at a remarkable inflection point in human history, one where the boundaries between the physical and digital realms are becoming increasingly blurred, "
                "where the speed of technological change is accelerating at an unprecedented rate, and where the decisions we make collectively over the next decade will shape "
                "the trajectory of civilization for generations to come. The challenges before us are immense and complex. Climate change threatens the stability of ecosystems "
                "that billions of people depend upon for their livelihoods and survival. Rapid urbanization is straining the infrastructure of cities around the world. "
                "Economic inequality continues to widen in many regions, creating social tensions that can destabilize even the most resilient of societies. "
                "And yet, I have never been more optimistic about our collective capacity to address these challenges. The tools available to us today, from artificial intelligence "
                "and advanced robotics to gene editing technologies and renewable energy systems, represent a remarkable arsenal of solutions waiting to be deployed at scale. "
                "What we need now, more than ever, is the wisdom to use these tools responsibly, the courage to challenge outdated assumptions and entrenched interests, "
                "and the collaborative spirit to work across borders, disciplines, and generations to build a future worthy of the best that humanity has to offer."
            ),
        ],
    }


def get_edge_cases() -> list[dict]:
    """~130 edge cases across 17 categories for Test 2.7."""
    cases = []

    # 1. Empty / whitespace
    for text in ["", " ", "   ", "\t", "\n"]:
        cases.append({"category": "empty_whitespace", "text": text})

    # 2. Single character / word
    for text in ["A", "I", "Oh", "Hi", "Yes", "No", "OK"]:
        cases.append({"category": "single_word", "text": text})

    # 3. Very long text (500+ words)
    long_text = " ".join(["The quick brown fox jumps over the lazy dog."] * 40)
    cases.append({"category": "very_long", "text": long_text})
    cases.append({"category": "very_long", "text": "A " * 500})

    # 4. Numbers and digits
    for text in [
        "1234567890",
        "The answer is 42.",
        "Call 1-800-555-0199 for assistance.",
        "Pi is approximately 3.14159265358979.",
        "The year 2024 saw significant AI advancements.",
        "Temperature: -40°C equals -40°F exactly.",
        "The ratio is 3:2 and the fraction is 7/8.",
        "$1,234.56 USD",
        "99.9% uptime guarantee",
        "Order #A-123-456-789",
    ]:
        cases.append({"category": "numbers", "text": text})

    # 5. Punctuation heavy
    for text in [
        "Hello, world! How are you?",
        "Wait... really? No way!",
        "Items: apples, bananas, cherries; and more.",
        "The result (as expected) was positive.",
        "She said: \"I'll be there at 5 o'clock.\"",
        "Note — this is important — please read carefully.",
        "E.g., i.e., etc. are common abbreviations.",
        "...",
        "!!!",
        "???",
    ]:
        cases.append({"category": "punctuation", "text": text})

    # 6. Abbreviations and acronyms
    for text in [
        "The CEO of the USA met with the UN Secretary-General.",
        "NASA's JPL is located in Pasadena, CA.",
        "Please send an RSVP by EOD on Friday.",
        "The API uses REST and JSON over HTTPS.",
        "Dr. Smith and Prof. Jones attended the MIT symposium.",
        "She holds a Ph.D. in AI from CMU.",
        "The NATO summit was held in Washington D.C.",
        "ETA: 15 min. ETD: 30 min. ETE: 45 min.",
    ]:
        cases.append({"category": "abbreviations", "text": text})

    # 7. Proper nouns
    for text in [
        "Barack Obama visited Beijing and met Xi Jinping.",
        "The Eiffel Tower is located in Paris, France.",
        "Shakespeare wrote Hamlet and A Midsummer Night's Dream.",
        "Tesla's Gigafactory is in Sparks, Nevada.",
        "Mount Kilimanjaro is the highest peak in Africa.",
        "The Thames flows through London to the North Sea.",
    ]:
        cases.append({"category": "proper_nouns", "text": text})

    # 8. SSML-like markup
    for text in [
        "<speak>Hello world</speak>",
        "Hello <break time='500ms'/> world",
        "<prosody rate='slow'>Speak slowly please</prosody>",
        "Text with <emphasis>important</emphasis> word.",
        "Normal text with <unknown_tag> inside.",
    ]:
        cases.append({"category": "ssml_markup", "text": text})

    # 9. Special characters / Unicode
    for text in [
        "Café au lait and crêpes are French staples.",
        "München ist eine schöne Stadt in Deutschland.",
        "The résumé was submitted to the naïve recruiter.",
        "Piñata fiesta at señor García's hacienda.",
        "The coöperation between naïve and résumé speakers.",
        "Emoji in text: great job 👍",
        "100% ™ ® © ± ≈ ∞",
    ]:
        cases.append({"category": "special_chars", "text": text})

    # 10. Repeated words / phrases
    for text in [
        "Very very very very very long sentence.",
        "The the the the the",
        "No no no no no no no!",
        "Help help help help help help help help help help.",
    ]:
        cases.append({"category": "repetition", "text": text})

    # 11. Mixed case
    for text in [
        "ALL CAPS SENTENCE HERE.",
        "all lowercase sentence here.",
        "mIxEd CaSe TeXt HerE.",
        "iPhone and macOS are Apple products.",
        "eBay, eCommerce, and eSports are popular.",
    ]:
        cases.append({"category": "mixed_case", "text": text})

    # 12. Technical / domain vocabulary
    for text in [
        "The hypoxia-inducible factor alpha regulates angiogenesis.",
        "Quantum entanglement enables non-local correlations between qubits.",
        "The Navier-Stokes equations describe fluid dynamics.",
        "CRISPR-Cas9 enables precise genomic editing at specific loci.",
        "The eigenvalue decomposition of the covariance matrix yields principal components.",
        "Photosynthesis converts CO2 and H2O to glucose and O2.",
    ]:
        cases.append({"category": "technical_domain", "text": text})

    # 13. Questions and commands
    for text in [
        "What time is it?",
        "Can you please help me?",
        "Where is the nearest hospital?",
        "Stop! Don't move!",
        "Please sit down and be quiet.",
        "Why did this happen?",
        "How much does it cost?",
    ]:
        cases.append({"category": "questions_commands", "text": text})

    # 14. Lists and enumerations
    for text in [
        "First, second, third, fourth, fifth.",
        "Steps: 1) Open the app. 2) Log in. 3) Click settings.",
        "Monday, Tuesday, Wednesday, Thursday, Friday.",
        "Red, orange, yellow, green, blue, indigo, violet.",
        "Alpha, Beta, Gamma, Delta, Epsilon, Zeta.",
    ]:
        cases.append({"category": "lists", "text": text})

    # 15. URLs and emails
    for text in [
        "Visit https://www.example.com for more information.",
        "Send an email to support@example.com.",
        "The file is at /home/user/documents/report.pdf",
        "Follow us @company on Twitter and LinkedIn.",
    ]:
        cases.append({"category": "urls_emails", "text": text})

    # 16. Code snippets
    for text in [
        "x = 5; y = 10; z = x + y",
        "if (x > 0) { return true; } else { return false; }",
        "SELECT * FROM users WHERE id = 42;",
        "print('Hello, World!')",
    ]:
        cases.append({"category": "code_snippets", "text": text})

    # 17. Borderline length (just under/over typical limits)
    cases.append({"category": "boundary_length", "text": "A" * 499})
    cases.append({"category": "boundary_length", "text": "Hello world. " * 38})
    cases.append({"category": "boundary_length", "text": "Short."})
    cases.append({"category": "boundary_length", "text": "The quick brown fox." * 10})

    return cases


def get_long_form_passages() -> list[dict]:
    """5 multi-paragraph passages for Test 2.8 long-form consistency."""
    return [
        {
            "title": "Technology and Society",
            "paragraphs": [
                "Artificial intelligence is rapidly reshaping every sector of the modern economy. From automated customer service agents to sophisticated medical diagnostic tools, the applications of machine learning continue to expand at a remarkable pace.",
                "The labor market implications are particularly significant. While some routine cognitive tasks are being automated, new categories of work are emerging that require distinctly human skills such as creativity, empathy, and complex judgment.",
                "Policymakers around the world are grappling with how to regulate these technologies effectively. The challenge lies in fostering innovation while ensuring that the benefits of AI are distributed equitably across society.",
                "Educational institutions are adapting their curricula to prepare students for an AI-augmented workplace. Computational thinking, data literacy, and human-computer collaboration are becoming as fundamental as traditional reading and writing skills.",
                "Despite the challenges, the potential of artificial intelligence to address humanity's greatest problems remains immense. From accelerating drug discovery to optimizing energy grids, the technology holds genuine promise for improving human wellbeing.",
            ],
        },
        {
            "title": "Climate and Environment",
            "paragraphs": [
                "The scientific consensus on climate change is unequivocal. The Earth's average surface temperature has risen by approximately one point one degrees Celsius since the pre-industrial era, driven primarily by the burning of fossil fuels.",
                "The consequences of this warming are already being felt across the globe. Rising sea levels threaten coastal communities, while more frequent and severe weather events are causing increasing economic damage and humanitarian suffering.",
                "The transition to renewable energy is accelerating, with solar and wind power now representing the cheapest sources of new electricity generation in most parts of the world. Battery storage technology is improving rapidly, addressing the intermittency challenge.",
                "Carbon capture and removal technologies are being developed at scale, though their costs remain high and their long-term viability is still being assessed. Nature-based solutions such as reforestation also offer significant potential for carbon sequestration.",
                "International cooperation remains essential. The Paris Agreement framework, despite its limitations, represents a crucial mechanism for coordinating global action. Achieving its goals will require unprecedented levels of ambition from all major economies.",
                "Individual choices also matter. Diet, transportation, housing, and consumption patterns all have significant carbon footprints. Consumer behavior change, supported by appropriate pricing signals and social norms, is an important complement to systemic action.",
            ],
        },
        {
            "title": "Medical Advances",
            "paragraphs": [
                "The development of messenger RNA vaccines during the COVID-19 pandemic represented a watershed moment in medical history. This platform technology, which had been in development for decades, proved capable of being adapted to new pathogens with unprecedented speed.",
                "Gene therapy is another area of rapid progress. Recent clinical trials have demonstrated the ability to correct genetic defects underlying conditions such as sickle cell disease and certain forms of blindness, offering hope to patients who previously had few treatment options.",
                "Precision medicine is transforming cancer treatment. By analyzing the specific genetic mutations driving an individual patient's tumor, oncologists can select targeted therapies that are far more effective and less toxic than traditional chemotherapy.",
                "Neuroscience is advancing our understanding of conditions such as Alzheimer's disease and depression. New diagnostic biomarkers and therapeutic targets are being identified, though translating these discoveries into effective treatments remains challenging.",
                "The integration of artificial intelligence into medical practice is enabling earlier and more accurate diagnosis across many specialties. Machine learning algorithms trained on large datasets of medical images can detect patterns invisible to the human eye.",
            ],
        },
        {
            "title": "Space Exploration",
            "paragraphs": [
                "The return of humans to deep space is now a concrete goal with defined timelines. NASA's Artemis program aims to land the first woman and first person of color on the lunar surface, establishing a sustainable presence that will serve as a proving ground for eventual Mars missions.",
                "Commercial space companies have fundamentally changed the economics of launch. Reusable rocket technology has reduced the cost of placing a kilogram into orbit by more than an order of magnitude compared to the Space Shuttle era.",
                "The James Webb Space Telescope has opened new windows onto the cosmos. Its infrared capabilities allow astronomers to observe the formation of the earliest galaxies, probe the atmospheres of exoplanets, and study stellar nurseries in unprecedented detail.",
                "Mars exploration continues to yield remarkable discoveries. Perseverance has collected rock samples that may contain evidence of ancient microbial life, while Ingenuity demonstrated the feasibility of powered flight in the thin Martian atmosphere.",
                "The search for life beyond Earth has intensified. Enceladus and Europa, moons of Saturn and Jupiter respectively, harbor subsurface oceans that may provide habitable conditions. Future missions are being designed to probe these environments directly.",
            ],
        },
        {
            "title": "Financial Markets",
            "paragraphs": [
                "Global equity markets have navigated a period of extraordinary volatility driven by pandemic disruptions, geopolitical tensions, and the fastest interest rate hiking cycle in four decades. Despite these headwinds, corporate earnings have proven resilient.",
                "The Federal Reserve's aggressive monetary tightening campaign succeeded in bringing inflation down from four-decade highs, though the path involved significant uncertainty about the risk of recession. The so-called soft landing outcome remained in view.",
                "Private credit markets have expanded dramatically as banks retreated from certain lending activities in response to tighter capital requirements. Alternative asset managers have filled this gap, raising concerns about systemic risk accumulation outside the regulated banking sector.",
                "Cryptocurrency markets experienced boom and bust cycles, with the collapse of several major exchanges and protocols highlighting the risks of unregulated digital asset markets. Regulatory clarity remains a work in progress across most jurisdictions.",
                "Environmental, social, and governance investing continued to grow in assets under management, though debate intensified about the empirical evidence for its financial efficacy and the consistency of ESG ratings across different providers.",
            ],
        },
    ]
