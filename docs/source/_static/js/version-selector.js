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

        // Use URL utilities from the parent script if available, or create our own
        const utils = window.urlUtils || {
            getGitHubInfo: function() {
                const isGitHubPages = window.location.hostname.includes('github.io');
                let repoName = '';

                if (isGitHubPages && window.location.pathname.split('/').length > 1) {
                    repoName = window.location.pathname.split('/')[1];
                }

                return { isGitHubPages, repoName };
            },

            getPathInfo: function() {
                const currentPath = window.location.pathname;
                const pathParts = currentPath.split('/');
                const { isGitHubPages, repoName } = this.getGitHubInfo();

                let currentVersion = '';
                let pathAfterVersion = '';
                const startIndex = isGitHubPages ? 2 : 1; // Skip repo name if on GitHub Pages

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

            createVersionUrl: function(version, pathAfterVersion) {
                const { isGitHubPages, repoName } = this.getGitHubInfo();
                const origin = window.location.origin;
                const basePath = isGitHubPages && repoName ? '/' + repoName + '/' : '/';

                return origin + basePath + version + '/' + (pathAfterVersion || '');
            }
        };

        // Get the current path to determine which option should be selected
        const { pathAfterVersion, currentVersion } = utils.getPathInfo();
        const path = window.location.pathname;

        // Add specific versions to dropdown
        versions.forEach(function(ver) {
            // Skip if this version already exists in the dropdown
            if (Array.from(dropdown.options).some(opt => opt.value === ver.path)) {
                return;
            }

            const option = document.createElement('option');

            // Use just the version identifier as value
            option.value = ver.path; // This will be something like "latest" or "dev" or "v1.2.3"
            option.textContent = 'v' + ver.version;

            // Add title attribute with full URL for native browser tooltip
            option.title = utils.createVersionUrl(ver.path, pathAfterVersion);

            // Set as selected if we're on this version
            if (path.includes('/' + ver.path + '/')) {
                option.selected = true;
            }

            dropdown.appendChild(option);
        });
    } catch (e) {
        console.error('Error parsing versions JSON:', e);
    }
});
