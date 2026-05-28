/**
 * AReaL Documentation Language Toggle
 * Provides EN/ZH language switching with localStorage preference
 * Implementation differs from slime to maintain uniqueness
 */

(function() {
  'use strict';

  var STORAGE_KEY = 'areal-doc-lang';
  var LANG_EN = 'en';
  var LANG_ZH = 'zh';

  /**
   * Extract language from current URL path
   * Supports: /en/, /zh/, /AReaL/en/, /AReaL/zh/
   */
  function getCurrentLanguage() {
    var path = window.location.pathname;
    var segments = path.split('/').filter(Boolean);

    // Check for language in first or second position
    if (segments[0] === LANG_EN || segments[0] === LANG_ZH) {
      return segments[0];
    }
    if (segments[1] === LANG_EN || segments[1] === LANG_ZH) {
      return segments[1];
    }
    return null;
  }

  /**
   * Determine repository root from URL
   * Returns: 'AReaL' for project site, or null for user site
   */
  function getRepoRoot() {
    var host = window.location.host;
    var segments = window.location.pathname.split('/').filter(Boolean);

    // GitHub Pages project site: ends with github.io and has repo name
    if (host.endsWith('github.io') && segments.length > 0) {
      // Check if first segment could be repo name (not 'en' or 'zh')
      if (segments[0] !== LANG_EN && segments[0] !== LANG_ZH) {
        return segments[0];
      }
    }
    return null;
  }

  /**
   * Build target URL for language switch
   */
  function buildTargetUrl(targetLang) {
    var currentLang = getCurrentLanguage();
    var repoRoot = getRepoRoot();
    var segments = window.location.pathname.split('/').filter(Boolean);
    var hasLang = segments[0] === LANG_EN || segments[0] === LANG_ZH ||
                  segments[1] === LANG_EN || segments[1] === LANG_ZH;

    if (hasLang) {
      // Replace existing language prefix
      if (segments[0] === LANG_EN || segments[0] === LANG_ZH) {
        segments[0] = targetLang;
      } else if (segments[1] === LANG_EN || segments[1] === LANG_ZH) {
        segments[1] = targetLang;
      }
    } else {
      // Insert language prefix
      if (repoRoot) {
        // Project site: insert after repo root
        segments.splice(1, 0, targetLang);
      } else {
        // User site: insert at beginning
        segments.unshift(targetLang);
      }
    }

    var newPath = '/' + segments.join('/');
    var url = new URL(window.location.href);
    url.pathname = newPath;
    // Remove trailing slash if it was originally a file (not directory)
    var originalPath = window.location.pathname;
    if (originalPath.endsWith('/') && !newPath.endsWith('/')) {
      newPath = newPath + '/';
    } else if (!originalPath.endsWith('/') && newPath.endsWith('/')) {
      newPath = newPath.replace(/\/$/, '');
    }
    url.pathname = newPath;
    return url.toString();
  }

  /**
   * Save language preference to localStorage
   */
  function savePreference(lang) {
    try {
      localStorage.setItem(STORAGE_KEY, lang);
    } catch (e) {
      console.warn('Could not save language preference to localStorage.', e);
    }
  }

  /**
   * Get size from existing header icons
   */
  function getIconSize() {
    // Try to find existing icons in the header
    var iconSelectors = [
      '.header-article-items__end .btn svg',
      '.header-article-items__end i',
      '.header-article-items__end .dropdown-toggle svg',
      '.bd-header .btn svg'
    ];

    for (var i = 0; i < iconSelectors.length; i++) {
      var icon = document.querySelector(iconSelectors[i]);
      if (icon) {
        // Get computed size
        var style = window.getComputedStyle(icon);
        var width = parseFloat(style.width);
        var height = parseFloat(style.height);
        if (width > 0 && height > 0) {
          return { width: width, height: height };
        }
        // Try getAttribute for svg
        var w = icon.getAttribute('width');
        var h = icon.getAttribute('height');
        if (w && h) {
          return { width: parseFloat(w), height: parseFloat(h) };
        }
      }
    }
    // Default size
    return { width: 16, height: 16 };
  }

  /**
   * Create and insert language toggle button
   */
  function createButton() {
    var currentLang = getCurrentLanguage();
    if (!currentLang) return;

    var targetLang = currentLang === LANG_EN ? LANG_ZH : LANG_EN;
    var buttonText = currentLang === LANG_EN ? '中文' : 'EN';
    var titleText = currentLang === LANG_EN ? 'Switch to Chinese' : 'Switch to English';

    // Get icon size from existing icons
    var iconSize = getIconSize();
    var fontSize = iconSize.width * 0.85;

    var button = document.createElement('button');
    button.id = 'areal-lang-toggle';
    button.className = 'areal-lang-toggle-btn';
    button.type = 'button';
    button.title = titleText;
    button.setAttribute('aria-label', titleText);
    button.style.fontSize = fontSize + 'px';

    // Add translate icon with dynamic size + text
    button.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="' + iconSize.width + '" height="' + iconSize.height + '" fill="currentColor" viewBox="0 0 16 16" class="areal-lang-icon"><path d="M4.545 6.714 4.11 8H3l1.862-5h1.284L8 8H6.833l-.435-1.286zm1.634-.736L5.5 3.956h-.049l-.679 2.022z"/><path d="M0 2a2 2 0 0 1 2-2h7a2 2 0 0 1 2 2v3h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2v-3H2a2 2 0 0 1-2-2zm2-1a1 1 0 0 0-1 1v7a1 1 0 0 0 1 1h7a1 1 0 0 0 1-1V2a1 1 0 0 0-1-1zm7.138 9.995q.289.451.63.846c-.748.575-1.673 1.001-2.768 1.292.178.217.451.635.555.867 1.125-.359 2.08-.844 2.886-1.494.777.665 1.739 1.165 2.93 1.472.133-.254.414-.673.629-.89-1.125-.253-2.057-.694-2.82-1.284.681-.747 1.222-1.651 1.621-2.757H14V8h-3v1.047h.765c-.318.844-.74 1.546-1.272 2.13a6 6 0 0 1-.415-.492 2 2 0 0 1-.94.31"/></svg><span class="areal-lang-text">' + buttonText + '</span>';

    button.addEventListener('click', function(e) {
      e.preventDefault();
      var targetUrl = buildTargetUrl(targetLang);
      savePreference(targetLang);
      window.location.href = targetUrl;
    });

    // Find appropriate container - append to header-article-items__end (right side of header)
    var container = document.querySelector('.header-article-items__end');

    if (container) {
      container.appendChild(button);
    }
  }

  /**
   * Initialize on page load
   */
  function init() {
    if (!getCurrentLanguage()) return;

    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', createButton);
    } else {
      createButton();
    }
  }

  init();
})();
