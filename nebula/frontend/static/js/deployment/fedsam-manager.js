// FedSAM Training Module
const FedSAMManager = (function() {
    const Settings = {
        enabled: false,
        rho: 0.5
    };

    function initialize() {
        const fedsamSwitch = document.getElementById('fedsamSwitch');
        const fedsamSettings = document.getElementById('fedsamSettings');
        const fedsamRho = document.getElementById('fedsamRho');

        if (!fedsamSwitch || !fedsamSettings || !fedsamRho) {
            console.warn('FedSAM controls not found in DOM');
            return;
        }

        // Toggle settings visibility when switch is changed
        fedsamSwitch.addEventListener('change', function() {
            Settings.enabled = fedsamSwitch.checked;
            fedsamSettings.style.display = fedsamSwitch.checked ? 'block' : 'none';
            console.log('FedSAM enabled:', Settings.enabled);
        });

        // Rho parameter listener
        fedsamRho.addEventListener('input', function() {
            Settings.rho = parseFloat(fedsamRho.value) || 0.5;
            console.log('FedSAM rho updated:', Settings.rho);
        });

        console.log('FedSAM controls initialized');
    }

    function getSettings() {
        return {
            enabled: Settings.enabled,
            rho: Settings.rho
        };
    }

    function isEnabled() {
        return Settings.enabled;
    }

    function getRho() {
        return Settings.rho;
    }

    function setSettings(settings) {
        if (settings) {
            Settings.enabled = settings.enabled || false;
            Settings.rho = settings.rho || 0.5;

            // Update UI
            const fedsamSwitch = document.getElementById('fedsamSwitch');
            const fedsamSettings = document.getElementById('fedsamSettings');
            const fedsamRho = document.getElementById('fedsamRho');

            if (fedsamSwitch) {
                fedsamSwitch.checked = Settings.enabled;
                if (fedsamSettings) {
                    fedsamSettings.style.display = Settings.enabled ? 'block' : 'none';
                }
            }
            if (fedsamRho) {
                fedsamRho.value = Settings.rho;
            }
        }
    }

    function reset() {
        Settings.enabled = false;
        Settings.rho = 0.5;

        const fedsamSwitch = document.getElementById('fedsamSwitch');
        const fedsamSettings = document.getElementById('fedsamSettings');
        const fedsamRho = document.getElementById('fedsamRho');

        if (fedsamSwitch) fedsamSwitch.checked = false;
        if (fedsamSettings) fedsamSettings.style.display = 'none';
        if (fedsamRho) fedsamRho.value = 0.5;
    }

    return {
        initialize,
        getSettings,
        isEnabled,
        getRho,
        setSettings,
        reset
    };
})();

export default FedSAMManager;
