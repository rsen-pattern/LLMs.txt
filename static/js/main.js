(function () {
    // Tab switching
    document.querySelectorAll('.tab-btn').forEach(function (btn) {
        btn.addEventListener('click', function () {
            var tab = btn.dataset.tab;
            document.querySelectorAll('.tab-btn').forEach(function (b) {
                b.classList.remove('active');
                b.setAttribute('aria-selected', 'false');
            });
            document.querySelectorAll('.tab-panel').forEach(function (p) {
                p.classList.remove('active');
                p.hidden = true;
            });
            btn.classList.add('active');
            btn.setAttribute('aria-selected', 'true');
            var panel = document.getElementById('tab-' + tab);
            if (panel) {
                panel.classList.add('active');
                panel.hidden = false;
            }
        });
    });

    // File upload zone label
    document.querySelectorAll('.file-zone input[type="file"]').forEach(function (input) {
        var zone = input.closest('.file-zone');
        var nameEl = zone && zone.querySelector('.file-zone-name');
        if (!nameEl) return;

        input.addEventListener('change', function () {
            if (input.files && input.files.length > 0) {
                zone.classList.add('has-file');
                nameEl.textContent = input.files[0].name;
            } else {
                zone.classList.remove('has-file');
                nameEl.textContent = '';
            }
        });
    });

    // Stagger flash messages
    document.querySelectorAll('.flash-stack .flash').forEach(function (el, i) {
        el.style.animationDelay = (i * 0.08) + 's';
    });
})();
