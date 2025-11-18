@echo off
REM Batch script to help convert HTML files to PDF using default browser
REM This opens each HTML file - you can then use browser Print > Save as PDF

echo Opening HTML files for PDF conversion...
echo.
echo Instructions:
echo 1. Each HTML file will open in your default browser
echo 2. Press Ctrl+P in the browser
echo 3. Select "Save as PDF" or "Microsoft Print to PDF"
echo 4. Save with the same name but .pdf extension
echo 5. Close the browser tab and press any key to continue to next file
echo.

cd attachments

start "" "technical_specification.html"
pause
start "" "architecture_diagrams.html"
pause
start "" "detailed_roadmap.html"
pause
start "" "performance_benchmarks.html"
pause
start "" "use_cases.html"
pause
start "" "comparison_matrix.html"
pause
start "" "community_engagement_plan.html"
pause

echo.
echo All files opened! Remember to save each as PDF.
echo PDFs should be saved in the attachments folder.
pause

