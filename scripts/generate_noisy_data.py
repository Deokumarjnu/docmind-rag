#!/usr/bin/env python3
"""Generate noisy PDF documents for testing RAG noise filtering."""

import os
import random
from pathlib import Path

# Try to import reportlab, if not available use text files
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    print("reportlab not installed, generating text files instead")

# Noisy content topics (completely unrelated to AI/ML papers)
NOISE_TOPICS = {
    "cooking": [
        "The art of French cuisine requires patience and precision.",
        "Baking sourdough bread starts with a healthy starter culture.",
        "Mediterranean diet emphasizes olive oil, vegetables, and fish.",
        "Spices like turmeric, cumin, and coriander add depth to curries.",
        "Fermentation is an ancient technique for preserving foods.",
        "The Maillard reaction creates the brown crust on seared meat.",
        "Emulsification is key to making mayonnaise and hollandaise sauce.",
        "Sous vide cooking allows precise temperature control.",
        "Knife skills are fundamental to efficient kitchen work.",
        "Mise en place means having all ingredients ready before cooking.",
    ],
    "gardening": [
        "Composting converts kitchen scraps into nutrient-rich soil.",
        "Pruning roses in late winter encourages spring growth.",
        "Companion planting helps control pests naturally.",
        "Raised beds improve drainage and soil quality.",
        "Mulching reduces water evaporation and suppresses weeds.",
        "Perennial plants return year after year with minimal care.",
        "pH levels affect nutrient availability in soil.",
        "Beneficial insects like ladybugs control aphid populations.",
        "Succession planting ensures continuous harvests.",
        "Native plants require less water and maintenance.",
    ],
    "history": [
        "The Roman Empire lasted over a thousand years.",
        "The Industrial Revolution transformed manufacturing.",
        "Ancient Egypt developed sophisticated irrigation systems.",
        "The Renaissance sparked cultural and artistic rebirth.",
        "World War I reshaped European borders and politics.",
        "The Silk Road connected East and West for centuries.",
        "Medieval castles served as defensive fortifications.",
        "The printing press revolutionized information sharing.",
        "Colonial expansion affected global demographics.",
        "Ancient Greek democracy influenced modern governments.",
    ],
    "sports": [
        "Basketball was invented by James Naismith in 1891.",
        "Marathon running covers 26.2 miles or 42.195 kilometers.",
        "Cricket matches can last up to five days.",
        "Swimming is an excellent low-impact cardiovascular exercise.",
        "Tennis scoring uses love, 15, 30, 40, and game.",
        "Soccer is the most popular sport worldwide.",
        "Golf courses typically have 18 holes.",
        "Olympic athletes train for years to compete.",
        "Baseball has nine innings in a regulation game.",
        "Cycling builds leg strength and endurance.",
    ],
    "travel": [
        "The Great Wall of China spans thousands of miles.",
        "Paris is known for the Eiffel Tower and Louvre Museum.",
        "Japan combines ancient traditions with modern technology.",
        "The Grand Canyon reveals millions of years of geology.",
        "Venice is built on 118 small islands in a lagoon.",
        "Machu Picchu sits high in the Peruvian Andes.",
        "Safari tours in Africa offer wildlife viewing.",
        "Iceland features geysers and volcanic landscapes.",
        "The Northern Lights are visible in polar regions.",
        "Backpacking is a popular way to explore on a budget.",
    ],
    "finance": [
        "Compound interest grows investments exponentially over time.",
        "Diversification reduces portfolio risk.",
        "Stock markets experience cycles of bull and bear phases.",
        "Bonds provide fixed income with lower risk than stocks.",
        "Emergency funds should cover three to six months of expenses.",
        "Tax-advantaged accounts help with retirement savings.",
        "Credit scores affect loan interest rates.",
        "Real estate can provide rental income and appreciation.",
        "Inflation erodes purchasing power over time.",
        "Dollar-cost averaging reduces market timing risk.",
    ],
    "automotive": [
        "Internal combustion engines convert fuel to motion.",
        "Electric vehicles use lithium-ion battery packs.",
        "Regular oil changes extend engine life.",
        "Tire pressure affects fuel efficiency and handling.",
        "Antilock braking systems prevent wheel lockup.",
        "Hybrid cars combine electric and gasoline power.",
        "Turbochargers increase engine power output.",
        "Transmission systems transfer power to wheels.",
        "Suspension components affect ride comfort.",
        "Catalytic converters reduce harmful emissions.",
    ],
    "healthcare": [
        "Vaccines have eradicated smallpox worldwide.",
        "Regular exercise reduces cardiovascular disease risk.",
        "Sleep is essential for physical and mental health.",
        "Antibiotics treat bacterial but not viral infections.",
        "Blood pressure should be monitored regularly.",
        "Hydration is important for all bodily functions.",
        "Mental health is as important as physical health.",
        "Preventive screenings catch diseases early.",
        "Balanced nutrition supports immune function.",
        "Stress management techniques improve wellbeing.",
    ],
    "music": [
        "The piano has 88 keys spanning seven octaves.",
        "Classical music follows sonata and symphony forms.",
        "Jazz originated in New Orleans in the early 1900s.",
        "Rock and roll emerged in the 1950s.",
        "Orchestras include strings, winds, brass, and percussion.",
        "Music theory explains harmony, melody, and rhythm.",
        "Digital audio workstations enable music production.",
        "Vinyl records are experiencing a resurgence.",
        "Opera combines singing, acting, and orchestral music.",
        "Folk music reflects cultural traditions and stories.",
    ],
    "architecture": [
        "Gothic cathedrals feature pointed arches and flying buttresses.",
        "Modernist buildings emphasize function over ornamentation.",
        "Sustainable architecture reduces environmental impact.",
        "Load-bearing walls support structural weight.",
        "Cantilevers extend structures beyond their support.",
        "Skyscrapers require deep foundations and steel frames.",
        "Passive solar design maximizes natural heating and cooling.",
        "Art Deco style flourished in the 1920s and 1930s.",
        "Brutalism features raw concrete surfaces.",
        "Green roofs provide insulation and habitat.",
    ],
}


def generate_random_content(num_paragraphs=20):
    """Generate random paragraphs from various topics."""
    paragraphs = []
    topics = list(NOISE_TOPICS.keys())
    
    for _ in range(num_paragraphs):
        topic = random.choice(topics)
        sentences = random.sample(NOISE_TOPICS[topic], min(5, len(NOISE_TOPICS[topic])))
        paragraph = " ".join(sentences)
        paragraphs.append(paragraph)
    
    return paragraphs


def create_pdf(filepath, title, num_pages=5):
    """Create a PDF with random noisy content."""
    c = canvas.Canvas(str(filepath), pagesize=letter)
    width, height = letter
    
    for page in range(num_pages):
        # Title on first page
        if page == 0:
            c.setFont("Helvetica-Bold", 16)
            c.drawString(1*inch, height - 1*inch, title)
            y_position = height - 1.5*inch
        else:
            y_position = height - 1*inch
        
        c.setFont("Helvetica", 10)
        
        # Generate random paragraphs for this page
        paragraphs = generate_random_content(random.randint(3, 6))
        
        for para in paragraphs:
            # Word wrap
            words = para.split()
            line = ""
            for word in words:
                test_line = f"{line} {word}".strip()
                if c.stringWidth(test_line, "Helvetica", 10) < width - 2*inch:
                    line = test_line
                else:
                    c.drawString(1*inch, y_position, line)
                    y_position -= 14
                    line = word
                    
                    if y_position < 1*inch:
                        c.showPage()
                        y_position = height - 1*inch
                        c.setFont("Helvetica", 10)
            
            if line:
                c.drawString(1*inch, y_position, line)
                y_position -= 20  # Extra space between paragraphs
            
            if y_position < 1*inch:
                c.showPage()
                y_position = height - 1*inch
                c.setFont("Helvetica", 10)
        
        if page < num_pages - 1:
            c.showPage()
    
    c.save()


def create_text_file(filepath, title, num_paragraphs=30):
    """Create a text file with random noisy content."""
    with open(filepath, 'w') as f:
        f.write(f"# {title}\n\n")
        paragraphs = generate_random_content(num_paragraphs)
        for para in paragraphs:
            f.write(f"{para}\n\n")


def main():
    output_dir = Path("noisy_data")
    output_dir.mkdir(exist_ok=True)
    
    # Generate 95 noisy documents (to mix with 5 true AI papers = 5% true, 95% noise)
    num_docs = 95
    
    # Document titles with random topics
    doc_templates = [
        ("Cooking Techniques Manual", "cooking"),
        ("Garden Planning Guide", "gardening"),
        ("World History Overview", "history"),
        ("Sports Training Handbook", "sports"),
        ("Travel Destinations Guide", "travel"),
        ("Personal Finance Basics", "finance"),
        ("Automotive Maintenance Guide", "automotive"),
        ("Healthcare Essentials", "healthcare"),
        ("Music Theory Fundamentals", "music"),
        ("Architecture Design Principles", "architecture"),
    ]
    
    print(f"Generating {num_docs} noisy documents...")
    
    for i in range(num_docs):
        template_idx = i % len(doc_templates)
        base_title, topic = doc_templates[template_idx]
        title = f"{base_title} - Volume {i // len(doc_templates) + 1}"
        
        # Randomize document size (3-15 pages)
        num_pages = random.randint(3, 15)
        
        if HAS_REPORTLAB:
            filename = f"noise_{i+1:03d}_{topic}.pdf"
            filepath = output_dir / filename
            create_pdf(filepath, title, num_pages)
        else:
            filename = f"noise_{i+1:03d}_{topic}.txt"
            filepath = output_dir / filename
            create_text_file(filepath, title, num_pages * 5)
        
        if (i + 1) % 10 == 0:
            print(f"  Created {i + 1}/{num_docs} documents...")
    
    print(f"\n✅ Generated {num_docs} noisy documents in {output_dir}/")
    
    # Count files
    pdf_count = len(list(output_dir.glob("*.pdf")))
    txt_count = len(list(output_dir.glob("*.txt")))
    print(f"   PDFs: {pdf_count}, Text files: {txt_count}")
    
    # Show total size
    total_size = sum(f.stat().st_size for f in output_dir.iterdir() if f.is_file())
    print(f"   Total size: {total_size / (1024*1024):.2f} MB")


if __name__ == "__main__":
    main()

