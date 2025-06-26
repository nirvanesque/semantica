// Custom JavaScript for SemantiCore Documentation

document.addEventListener('DOMContentLoaded', function() {
    
    // Initialize copy buttons
    initializeCopyButtons();
    
    // Initialize search functionality
    initializeSearch();
    
    // Initialize smooth scrolling
    initializeSmoothScrolling();
    
    // Initialize code highlighting
    initializeCodeHighlighting();
    
    // Initialize mobile menu
    initializeMobileMenu();
    
    // Initialize dark mode toggle
    initializeDarkMode();
    
    // Initialize analytics
    initializeAnalytics();
});

// Copy button functionality
function initializeCopyButtons() {
    const copyButtons = document.querySelectorAll('.copybutton');
    
    copyButtons.forEach(button => {
        button.addEventListener('click', function() {
            const codeBlock = this.parentElement.querySelector('pre');
            const text = codeBlock.textContent;
            
            navigator.clipboard.writeText(text).then(() => {
                // Show success message
                const originalText = this.textContent;
                this.textContent = 'Copied!';
                this.style.backgroundColor = '#27AE60';
                
                setTimeout(() => {
                    this.textContent = originalText;
                    this.style.backgroundColor = '';
                }, 2000);
            }).catch(err => {
                console.error('Failed to copy text: ', err);
                // Fallback for older browsers
                fallbackCopyTextToClipboard(text, this);
            });
        });
    });
}

// Fallback copy function for older browsers
function fallbackCopyTextToClipboard(text, button) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.left = '-999999px';
    textArea.style.top = '-999999px';
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    
    try {
        document.execCommand('copy');
        const originalText = button.textContent;
        button.textContent = 'Copied!';
        button.style.backgroundColor = '#27AE60';
        
        setTimeout(() => {
            button.textContent = originalText;
            button.style.backgroundColor = '';
        }, 2000);
    } catch (err) {
        console.error('Fallback: Oops, unable to copy', err);
    }
    
    document.body.removeChild(textArea);
}

// Search functionality
function initializeSearch() {
    const searchInput = document.querySelector('.wy-side-nav-search input[type="text"]');
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            const query = this.value.toLowerCase();
            const menuItems = document.querySelectorAll('.wy-menu-vertical li');
            
            menuItems.forEach(item => {
                const link = item.querySelector('a');
                if (link) {
                    const text = link.textContent.toLowerCase();
                    if (text.includes(query)) {
                        item.style.display = '';
                    } else {
                        item.style.display = 'none';
                    }
                }
            });
        });
    }
}

// Smooth scrolling for anchor links
function initializeSmoothScrolling() {
    const links = document.querySelectorAll('a[href^="#"]');
    
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Code highlighting
function initializeCodeHighlighting() {
    // Add line numbers to code blocks
    const codeBlocks = document.querySelectorAll('pre code');
    
    codeBlocks.forEach(block => {
        const lines = block.textContent.split('\n');
        if (lines.length > 1) {
            const lineNumbers = document.createElement('div');
            lineNumbers.className = 'line-numbers';
            
            lines.forEach((line, index) => {
                const lineNumber = document.createElement('span');
                lineNumber.textContent = index + 1;
                lineNumber.className = 'line-number';
                lineNumbers.appendChild(lineNumber);
            });
            
            block.parentElement.insertBefore(lineNumbers, block);
        }
    });
}

// Mobile menu functionality
function initializeMobileMenu() {
    const menuToggle = document.querySelector('.wy-nav-top .wy-menu-toggle');
    const navSide = document.querySelector('.wy-nav-side');
    
    if (menuToggle && navSide) {
        menuToggle.addEventListener('click', function() {
            navSide.classList.toggle('nav-open');
        });
        
        // Close menu when clicking outside
        document.addEventListener('click', function(e) {
            if (!navSide.contains(e.target) && !menuToggle.contains(e.target)) {
                navSide.classList.remove('nav-open');
            }
        });
    }
}

// Dark mode toggle
function initializeDarkMode() {
    const darkModeToggle = document.createElement('button');
    darkModeToggle.className = 'dark-mode-toggle';
    darkModeToggle.innerHTML = '🌙';
    darkModeToggle.title = 'Toggle dark mode';
    
    // Check for saved dark mode preference
    const darkMode = localStorage.getItem('darkMode');
    if (darkMode === 'enabled') {
        document.body.classList.add('dark-mode');
        darkModeToggle.innerHTML = '☀️';
    }
    
    darkModeToggle.addEventListener('click', function() {
        document.body.classList.toggle('dark-mode');
        
        if (document.body.classList.contains('dark-mode')) {
            localStorage.setItem('darkMode', 'enabled');
            this.innerHTML = '☀️';
        } else {
            localStorage.setItem('darkMode', null);
            this.innerHTML = '🌙';
        }
    });
    
    // Add toggle button to navigation
    const navTop = document.querySelector('.wy-nav-top');
    if (navTop) {
        navTop.appendChild(darkModeToggle);
    }
}

// Analytics (Google Analytics 4)
function initializeAnalytics() {
    // Only load analytics in production
    if (window.location.hostname === 'semanticore.readthedocs.io') {
        // Google Analytics 4
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', 'G-XXXXXXXXXX'); // Replace with actual GA4 ID
        
        // Load Google Analytics script
        const script = document.createElement('script');
        script.async = true;
        script.src = 'https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX'; // Replace with actual GA4 ID
        document.head.appendChild(script);
        
        // Track page views
        gtag('config', 'G-XXXXXXXXXX', {
            page_title: document.title,
            page_location: window.location.href
        });
    }
}

// Progress bar for reading
function initializeProgressBar() {
    const progressBar = document.createElement('div');
    progressBar.className = 'reading-progress';
    progressBar.innerHTML = '<div class="progress-fill"></div>';
    document.body.appendChild(progressBar);
    
    window.addEventListener('scroll', function() {
        const scrollTop = window.pageYOffset;
        const docHeight = document.body.scrollHeight - window.innerHeight;
        const scrollPercent = (scrollTop / docHeight) * 100;
        
        const progressFill = progressBar.querySelector('.progress-fill');
        progressFill.style.width = scrollPercent + '%';
    });
}

// Table of contents highlighting
function initializeTOCHighlighting() {
    const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
    const tocLinks = document.querySelectorAll('.wy-menu-vertical a[href^="#"]');
    
    window.addEventListener('scroll', function() {
        let current = '';
        
        headings.forEach(heading => {
            const sectionTop = heading.offsetTop;
            const sectionHeight = heading.clientHeight;
            
            if (window.pageYOffset >= sectionTop - 200) {
                current = heading.getAttribute('id');
            }
        });
        
        tocLinks.forEach(link => {
            link.classList.remove('current');
            if (link.getAttribute('href') === '#' + current) {
                link.classList.add('current');
            }
        });
    });
}

// Keyboard shortcuts
function initializeKeyboardShortcuts() {
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + K for search
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            const searchInput = document.querySelector('.wy-side-nav-search input[type="text"]');
            if (searchInput) {
                searchInput.focus();
            }
        }
        
        // Escape to close mobile menu
        if (e.key === 'Escape') {
            const navSide = document.querySelector('.wy-nav-side');
            if (navSide) {
                navSide.classList.remove('nav-open');
            }
        }
        
        // Ctrl/Cmd + / for toggle dark mode
        if ((e.ctrlKey || e.metaKey) && e.key === '/') {
            e.preventDefault();
            const darkModeToggle = document.querySelector('.dark-mode-toggle');
            if (darkModeToggle) {
                darkModeToggle.click();
            }
        }
    });
}

// Initialize additional features
document.addEventListener('DOMContentLoaded', function() {
    initializeProgressBar();
    initializeTOCHighlighting();
    initializeKeyboardShortcuts();
});

// Performance monitoring
function initializePerformanceMonitoring() {
    // Monitor page load time
    window.addEventListener('load', function() {
        const loadTime = performance.timing.loadEventEnd - performance.timing.navigationStart;
        console.log('Page load time:', loadTime + 'ms');
        
        // Send to analytics if available
        if (typeof gtag !== 'undefined') {
            gtag('event', 'timing_complete', {
                name: 'load',
                value: loadTime
            });
        }
    });
    
    // Monitor scroll performance
    let scrollTimeout;
    window.addEventListener('scroll', function() {
        clearTimeout(scrollTimeout);
        scrollTimeout = setTimeout(function() {
            // Log scroll events for performance analysis
            console.log('Scroll event processed');
        }, 100);
    });
}

// Initialize performance monitoring
initializePerformanceMonitoring(); 