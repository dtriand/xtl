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

        // Get the current path to determine which option should be selected
        const currentPath = window.location.pathname;

        // Add specific versions to dropdown
        versions.forEach(function(ver) {
            const option = document.createElement('option');
            option.value = './' + ver.path;  // Use relative path with './'
            option.textContent = 'v' + ver.version;

            // Set as selected if we're on this version (check both absolute and relative paths)
            if (currentPath.includes('/' + ver.path)) {
                option.selected = true;
            }

            dropdown.appendChild(option);
        });
    } catch (e) {
        console.error('Error parsing versions JSON:', e);
    }
});
