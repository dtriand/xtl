/**
 * Version selector shared utilities for XTL documentation
 */

// Utility functions for URL handling
const urlUtils = {
    // Get GitHub Pages info (if applicable)
    getGitHubInfo: function() {
        const isGitHubPages = window.location.hostname.includes('github.io');
        let repoName = '';

        if (isGitHubPages && window.location.pathname.split('/').length > 1) {
            repoName = window.location.pathname.split('/')[1];
        }

        return { isGitHubPages, repoName };
    },

    // Extract path information from current URL
    getPathInfo: function() {
        const currentPath = window.location.pathname;
        const pathParts = currentPath.split('/');
        const { isGitHubPages, repoName } = this.getGitHubInfo();

        let currentVersion = '';
        let pathAfterVersion = '';
        const startIndex = isGitHubPages ? 2 : 1; // Skip repo name if on GitHub Pages

        // Find version segment in path
        for (let i = startIndex; i < pathParts.length; i++) {
            if (pathParts[i] === 'latest' ||
                pathParts[i] === 'dev' ||
                /^v?\d+(\.\d+)*$/.test(pathParts[i])) {
                currentVersion = pathParts[i];
                pathAfterVersion = pathParts.slice(i + 1).join('/');
                break;
            }
        }

        return { currentVersion, pathAfterVersion, repoName, isGitHubPages };
    },

    // Create a URL for a specific version
    createVersionUrl: function(version, pathAfterVersion) {
        const { isGitHubPages, repoName } = this.getGitHubInfo();
        const origin = window.location.origin;
        const basePath = isGitHubPages && repoName ? '/' + repoName + '/' : '/';

        return origin + basePath + version + '/' + (pathAfterVersion || '');
    },

    // Get base URL for the documentation site
    getBaseUrl: function() {
        const { isGitHubPages, repoName } = this.getGitHubInfo();
        const origin = window.location.origin;
        return origin + (isGitHubPages && repoName ? '/' + repoName : '');
    }
};

// Switch to a different documentation version (for dropdown)
function switchVersion(version) {
    if (!version) return;

    const { currentVersion, pathAfterVersion } = urlUtils.getPathInfo();

    // If we couldn't detect a version in the path, go to the version's root
    if (!currentVersion) {
        window.location.href = urlUtils.createVersionUrl(version, '');
        return;
    }

    // Construct the new URL with the selected version but same page path
    window.location.href = urlUtils.createVersionUrl(version, pathAfterVersion);
}

// Fetch versions.json and populate the dropdown (for documentation pages)
async function fetchVersionsForNav() {
    try {
        const baseUrl = urlUtils.getBaseUrl();
        const response = await fetch(`${baseUrl}/versions.json`);

        if (!response.ok) {
            console.error('Failed to fetch versions.json');
            return;
        }

        const versions = await response.json();
        populateVersionSelector(versions);
    } catch (error) {
        console.error('Error fetching versions:', error);
    }
}

// Populate the version selector dropdown (for documentation pages)
function populateVersionSelector(versions) {
    const dropdown = document.getElementById('nav-version-dropdown');
    if (!dropdown) return;

    // Clear existing options (except Latest and Dev)
    while (dropdown.options.length > 2) {
        dropdown.remove(2);
    }

    // Add version options
    versions.forEach(version => {
        // Skip adding if the option with the same value already exists
        if (Array.from(dropdown.options).some(opt => opt.value === version.path.replace('/', ''))) {
            return;
        }

        const option = document.createElement('option');
        option.value = version.path.replace('/', '');
        option.text = version.version;
        dropdown.appendChild(option);
    });

    // Set the current version in the dropdown
    const { currentVersion } = urlUtils.getPathInfo();
    if (currentVersion) {
        Array.from(dropdown.options).forEach(option => {
            if (option.value === currentVersion) {
                option.selected = true;
            }
        });
    }
}

// Add version selector to header buttons (for documentation pages)
function addVersionSelectorToNav() {
    const headerButtons = document.querySelector('.article-header-buttons');
    if (!headerButtons) return;

    // Create and insert version selector
    const versionSelector = document.createElement('div');
    versionSelector.className = 'version-selector-nav';

    versionSelector.innerHTML = '<select id="nav-version-dropdown" title="Version" ' +
        'onchange="switchVersion(this.value)">' +
        '<option value="latest">Latest</option>' +
        '<option value="dev">Dev</option></select>';

    headerButtons.insertBefore(versionSelector, headerButtons.firstChild);

    // Fetch versions and populate the dropdown
    fetchVersionsForNav();
}

// Functions for the root index page

// Fetch versions.json and handle redirection (for root index)
async function fetchVersionsAndRedirect() {
    try {
        const baseUrl = urlUtils.getBaseUrl();

        // First try to fetch versions.json
        const response = await fetch(`${baseUrl}/versions.json`);

        if (!response.ok) {
            console.error('Failed to fetch versions.json, trying direct redirects');
            tryDirectRedirect();
            return;
        }

        const versions = await response.json();

        // Check if we have versions and set up redirection
        if (versions && versions.length > 0) {
            // Try to redirect to latest first
            tryRedirectWithVersions(versions);

            // Populate version links
            populateVersionLinks(versions);
        } else {
            // No versions in the JSON, fall back to direct checks
            tryDirectRedirect();
        }
    } catch (error) {
        console.error('Error handling versions:', error);
        tryDirectRedirect();
    }
}

// Try to redirect using available versions from versions.json (for root index)
function tryRedirectWithVersions(versions) {
    const baseUrl = urlUtils.getBaseUrl();

    // Try latest first
    fetch(`${baseUrl}/latest/index.html`, { method: 'HEAD' })
        .then(response => {
            if (response.ok) {
                // Latest version exists, redirect
                window.location.href = `${baseUrl}/latest/`;
            } else {
                // Try dev version
                return fetch(`${baseUrl}/dev/index.html`, { method: 'HEAD' });
            }
        })
        .then(response => {
            if (response && response.ok) {
                // Dev version exists, redirect
                window.location.href = `${baseUrl}/dev/`;
            } else if (versions.length > 0) {
                // Try the newest specific version
                const newestVersion = versions[0];
                window.location.href = `${baseUrl}/${newestVersion.path}`;
            }
        })
        .catch(error => {
            console.error('Error during redirection:', error);
        });
}

// Direct redirection method (fallback for root index)
function tryDirectRedirect() {
    const baseUrl = urlUtils.getBaseUrl();

    // First try latest
    fetch(`${baseUrl}/latest/index.html`, { method: 'HEAD' })
        .then(response => {
            if (response.ok) {
                // Latest version exists, redirect
                window.location.href = `${baseUrl}/latest/`;
            } else {
                // Try dev version
                return fetch(`${baseUrl}/dev/index.html`, { method: 'HEAD' });
            }
        })
        .then(response => {
            if (response && response.ok) {
                // Dev version exists, redirect
                window.location.href = `${baseUrl}/dev/`;
            }
            // If we get here, neither version exists, so stay on the current page
        })
        .catch(error => {
            // Error occurred, stay on the current page
            console.error('Error checking documentation versions:', error);
        });
}

// Populate version links dynamically (for root index)
function populateVersionLinks(versions) {
    const baseUrl = urlUtils.getBaseUrl();
    const linksContainer = document.getElementById('version-links');
    if (!linksContainer) return;

    // Clear any existing content
    linksContainer.innerHTML = '';

    // Add the latest and dev links first
    linksContainer.innerHTML += `<a href="${baseUrl}/latest/">Latest Version</a> | `;
    linksContainer.innerHTML += `<a href="${baseUrl}/dev/">Development Version</a>`;

    // Add specific version links
    if (versions.length > 0) {
        linksContainer.innerHTML += '<hr>Specific versions:<br>';

        // Group versions by major version
        const versionGroups = {};
        versions.forEach(version => {
            const majorVersion = version.version.split('.')[0];
            if (!versionGroups[majorVersion]) {
                versionGroups[majorVersion] = [];
            }
            versionGroups[majorVersion].push(version);
        });

        // Add links for each major version group
        Object.keys(versionGroups).sort((a, b) => parseInt(b) - parseInt(a)).forEach(majorVersion => {
            const group = versionGroups[majorVersion];

            // Create a container for this major version
            const majorVersionDiv = document.createElement('div');
            majorVersionDiv.className = 'version-group';
            majorVersionDiv.innerHTML = `<strong>Version ${majorVersion}.x:</strong> `;

            // Add links for each version in this group
            group.sort((a, b) => {
                const partsA = a.version.split('.').map(Number);
                const partsB = b.version.split('.').map(Number);

                for (let i = 0; i < Math.max(partsA.length, partsB.length); i++) {
                    const partA = i < partsA.length ? partsA[i] : 0;
                    const partB = i < partsB.length ? partsB[i] : 0;
                    if (partA !== partB) return partB - partA;
                }
                return 0;
            }).forEach((version, index) => {
                if (index > 0) {
                    majorVersionDiv.innerHTML += ' | ';
                }
                majorVersionDiv.innerHTML += `<a href="${baseUrl}/${version.path}">${version.version}</a>`;
            });

            linksContainer.appendChild(majorVersionDiv);
        });
    }
}
