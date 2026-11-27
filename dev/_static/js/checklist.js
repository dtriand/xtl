/**
 * Checkboxes for the items of lists marked with `.. rst-class:: checklist`
 */

document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('ul.checklist > li').forEach(function(item) {
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.className = 'checklist-box';
        item.prepend(checkbox);
    });
});
