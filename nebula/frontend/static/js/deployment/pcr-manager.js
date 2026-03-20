// PCR (Predicted Consensus Regularization) Module
const PCRManager = (function() {
    const Settings = {
        enabled: false,
        mu: 0.01,
        apply_mode: 'pseudo_only'
    };

    function initialize() {
        const pcrSwitch = document.getElementById('pcrSwitch');
        const pcrSettings = document.getElementById('pcrSettings');
        const pcrMu = document.getElementById('pcrMu');
        const pcrApplyMode = document.getElementById('pcrApplyMode');

        if (!pcrSwitch || !pcrSettings || !pcrMu || !pcrApplyMode) {
            console.warn('PCR controls not found in DOM');
            return;
        }

        pcrSwitch.addEventListener('change', function() {
            Settings.enabled = pcrSwitch.checked;
            pcrSettings.style.display = pcrSwitch.checked ? 'block' : 'none';
            console.log('PCR enabled:', Settings.enabled);
        });

        pcrMu.addEventListener('input', function() {
            Settings.mu = parseFloat(pcrMu.value) || 0.01;
            console.log('PCR mu updated:', Settings.mu);
        });

        pcrApplyMode.addEventListener('change', function() {
            Settings.apply_mode = pcrApplyMode.value;
            console.log('PCR apply_mode updated:', Settings.apply_mode);
        });

        console.log('PCR controls initialized');
    }

    function getSettings() {
        return {
            enabled: Settings.enabled,
            mu: Settings.mu,
            apply_mode: Settings.apply_mode
        };
    }

    function isEnabled() {
        return Settings.enabled;
    }

    function setSettings(settings) {
        if (settings) {
            Settings.enabled = settings.enabled || false;
            Settings.mu = settings.mu || 0.01;
            Settings.apply_mode = settings.apply_mode || 'pseudo_only';

            const pcrSwitch = document.getElementById('pcrSwitch');
            const pcrSettings = document.getElementById('pcrSettings');
            const pcrMu = document.getElementById('pcrMu');
            const pcrApplyMode = document.getElementById('pcrApplyMode');

            if (pcrSwitch) {
                pcrSwitch.checked = Settings.enabled;
                if (pcrSettings) {
                    pcrSettings.style.display = Settings.enabled ? 'block' : 'none';
                }
            }
            if (pcrMu) pcrMu.value = Settings.mu;
            if (pcrApplyMode) pcrApplyMode.value = Settings.apply_mode;
        }
    }

    function reset() {
        Settings.enabled = false;
        Settings.mu = 0.01;
        Settings.apply_mode = 'pseudo_only';

        const pcrSwitch = document.getElementById('pcrSwitch');
        const pcrSettings = document.getElementById('pcrSettings');
        const pcrMu = document.getElementById('pcrMu');
        const pcrApplyMode = document.getElementById('pcrApplyMode');

        if (pcrSwitch) pcrSwitch.checked = false;
        if (pcrSettings) pcrSettings.style.display = 'none';
        if (pcrMu) pcrMu.value = 0.01;
        if (pcrApplyMode) pcrApplyMode.value = 'pseudo_only';
    }

    return {
        initialize,
        getSettings,
        isEnabled,
        setSettings,
        reset
    };
})();

export default PCRManager;
