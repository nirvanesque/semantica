#!/usr/bin/env python3
"""
Simple HTML to PDF converter using weasyprint.
Run: pip install weasyprint
Then: python convert_to_pdf_simple.py
"""

import os
from pathlib import Path

try:
    from weasyprint import HTML
    print("weasyprint is available!")
except ImportError:
    print("ERROR: weasyprint not installed.")
    print("Please install it first: pip install weasyprint")
    print("\nAlternatively, you can convert HTML to PDF manually:")
    print("1. Open each HTML file in a web browser")
    print("2. Press Ctrl+P (Print)")
    print("3. Select 'Save as PDF' as the destination")
    print("4. Save the PDF in the attachments folder")
    exit(1)

def convert_html_to_pdf(html_path, pdf_path):
    """Convert HTML file to PDF."""
    try:
        print(f"Converting {html_path.name}...")
        HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        size_kb = pdf_path.stat().st_size / 1024
        print(f"  ✓ Created {pdf_path.name} ({size_kb:.1f} KB)")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    attachments_dir = Path("attachments")
    if not attachments_dir.exists():
        print("attachments directory not found!")
        return
    
    html_files = list(attachments_dir.glob("*.html"))
    if not html_files:
        print("No HTML files found in attachments directory")
        return
    
    print(f"Found {len(html_files)} HTML files to convert\n")
    
    converted = 0
    for html_file in html_files:
        pdf_file = html_file.with_suffix('.pdf')
        if convert_html_to_pdf(html_file, pdf_file):
            converted += 1
    
    print(f"\n{'='*50}")
    print(f"Successfully converted {converted}/{len(html_files)} files to PDF")
    
    if converted == len(html_files):
        print("\nAll files converted! PDFs are ready for grant application.")
    else:
        print(f"\n{len(html_files) - converted} files failed to convert.")
        print("You can convert them manually using browser Print > Save as PDF")

if __name__ == "__main__":
    main()

