// Pseudo Aggregation Module
const PseudoAggManager = (function() {
    const Settings = {
        enabled: false,
        emaAlpha: 0.25
    };

    function initialize() {
        const pseudoAggSwitch = document.getElementById('pseudoAggSwitch');
        const pseudoAggSettings = document.getElementById('pseudoAggSettings');
        const emaAlphaInput = document.getElementById('emaAlphaInput');
        const emaAlphaValue = document.getElementById('emaAlphaValue');

        if (!pseudoAggSwitch || !pseudoAggSettings || !emaAlphaInput || !emaAlphaValue) {
            console.warn('Pseudo Aggregation controls not found in DOM');
            return;
        }

        // Toggle settings visibility when switch is changed
        pseudoAggSwitch.addEventListener('change', function() {
            Settings.enabled = pseudoAggSwitch.checked;
            pseudoAggSettings.style.display = pseudoAggSwitch.checked ? 'block' : 'none';
            console.log('Pseudo Aggregation enabled:', Settings.enabled);
        });

        // Sync slider with number input
        emaAlphaInput.addEventListener('input', function() {
            const value = parseFloat(emaAlphaInput.value) / 100; // Convert 0-100 to 0.0-1.0
            emaAlphaValue.value = value.toFixed(2);
            Settings.emaAlpha = value;
            console.log('EMA Alpha updated:', Settings.emaAlpha);
        });

        // Sync number input with slider
        emaAlphaValue.addEventListener('input', function() {
            const value = parseFloat(emaAlphaValue.value);
            if (value >= 0 && value <= 1) {
                emaAlphaInput.value = Math.round(value * 100); // Convert 0.0-1.0 to 0-100
                Settings.emaAlpha = value;
                console.log('EMA Alpha updated:', Settings.emaAlpha);
            }
        });

        console.log('Pseudo Aggregation controls initialized');
    }

    function getSettings() {
        return {
            enabled: Settings.enabled,
            ema_alpha: Settings.emaAlpha
        };
    }

    function isEnabled() {
        return Settings.enabled;
    }

    function getEmaAlpha() {
        return Settings.emaAlpha;
    }

    return {
        initialize,
        getSettings,
        isEnabled,
        getEmaAlpha
    };
})();

export default PseudoAggManager;
