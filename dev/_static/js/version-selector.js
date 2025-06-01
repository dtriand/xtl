/**
 * XTL Documentation Version Selector
 *
 * This script adds version options to the floating version selector
 */

document.addEventListener('DOMContentLoaded', function() {
    // Get the dropdown element
    const dropdown = document.getElementById('version-dropdown');
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

        // Helper function to get the current path after version
        function getCurrentPathAfterVersion() {
            const currentPath = window.location.pathname;
            const pathParts = currentPath.split('/');
            let pathAfterVersion = '';

            for (let i = 0; i < pathParts.length; i++) {
                if (pathParts[i] === 'latest' || pathParts[i] === 'dev' || /^v?\d+(\.\d+)*$/.test(pathParts[i])) {
                    pathAfterVersion = pathParts.slice(i + 1).join('/');
                    break;
                }
            }

            return pathAfterVersion;
        }

        // Helper function to create tooltip text for version options
        function createTooltipText(version, pathAfterVersion) {
            const origin = window.location.origin;
            return origin + '/' + version + '/' + (pathAfterVersion || '');
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

            // Add tooltip showing full URL path
            option.title = createTooltipText(ver.path, pathAfterVersion);

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
