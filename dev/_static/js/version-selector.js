/**
 * XTL Documentation Version Selector
 *
 * This script adds version options to the dropdown in the header
 */

document.addEventListener('DOMContentLoaded', function() {
    // Get the dropdown element
    const dropdown = document.getElementById('nav-version-dropdown');
    if (!dropdown) return;

    // Parse versions data
    try {
        let versions = [];

        // Check if the versionsJson variable exists and has been replaced
        if (typeof versionsJson !== 'undefined' && versionsJson && versionsJson !== '__VERSIONS_JSON__') {
            versions = JSON.parse(versionsJson);
        }

        if (!Array.isArray(versions) || versions.length === 0) {
            return;
        }

        // Helper function to detect GitHub Pages and get repo name
        function getGitHubPagesInfo() {
            const isGitHubPages = window.location.hostname.includes('github.io');
            let repoName = '';

            if (isGitHubPages) {
                const pathParts = window.location.pathname.split('/');
                if (pathParts.length > 1) {
                    repoName = pathParts[1];
                }
            }

            return { isGitHubPages, repoName };
        }

        // Helper function to get the current path after version
        function getCurrentPathAfterVersion() {
            const currentPath = window.location.pathname;
            const pathParts = currentPath.split('/');
            let pathAfterVersion = '';

            const { isGitHubPages } = getGitHubPagesInfo();
            const startIndex = isGitHubPages ? 2 : 1; // Skip repo name if on GitHub Pages

            for (let i = startIndex; i < pathParts.length; i++) {
                if (pathParts[i] === 'latest' || pathParts[i] === 'dev' || /^v?\d+(\.\d+)*$/.test(pathParts[i])) {
                    pathAfterVersion = pathParts.slice(i + 1).join('/');
                    break;
                }
            }

            return pathAfterVersion;
        }

        // Helper function to create URL for version options
        function createVersionUrl(version, pathAfterVersion) {
            const origin = window.location.origin;
            const { isGitHubPages, repoName } = getGitHubPagesInfo();

            if (isGitHubPages && repoName) {
                return origin + '/' + repoName + '/' + version + '/' + (pathAfterVersion || '');
            } else {
                return origin + '/' + version + '/' + (pathAfterVersion || '');
            }
        }

        // Get the current path to determine which option should be selected
        const currentPath = window.location.pathname;
        const pathAfterVersion = getCurrentPathAfterVersion();

        // Add specific versions to dropdown
        versions.forEach(function(ver) {
            const option = document.createElement('option');
            // Use just the version identifier as value
            option.value = ver.path; // This will be something like "latest" or "dev" or "v1.2.3"
            option.textContent = 'v' + ver.version;

            // Add title attribute with full URL for native browser tooltip
            const fullUrl = createVersionUrl(ver.path, pathAfterVersion);
            option.title = fullUrl;

            // Set as selected if we're on this version
            if (currentPath.includes('/' + ver.path + '/')) {
                option.selected = true;
            }

            dropdown.appendChild(option);
        });
    } catch (e) {
        console.error('Error parsing versions JSON:', e);
    }
});
