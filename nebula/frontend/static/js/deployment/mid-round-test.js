// Mid-Round Testing Manager Module
const MidRoundTestManager = (function() {
    const Settings = {
        enabled: false
    };

    function initialize() {
        const midRoundTestSwitch = document.getElementById('midRoundTestSwitch');
        const midRoundTestInfo = document.getElementById('midRoundTestInfo');
        const pseudoAggSwitch = document.getElementById('pseudoAggSwitch');

        if (!midRoundTestSwitch || !midRoundTestInfo) {
            console.error('Mid-round test elements not found');
            return;
        }

        // Toggle info visibility
        midRoundTestSwitch.addEventListener('change', function() {
            Settings.enabled = midRoundTestSwitch.checked;
            midRoundTestInfo.style.display = midRoundTestSwitch.checked ? 'block' : 'none';

            // Disable pseudo agg if mid-round test is enabled (they're mutually exclusive)
            if (midRoundTestSwitch.checked && pseudoAggSwitch && pseudoAggSwitch.checked) {
                alert('Mid-Round Testing and Pseudo Aggregation cannot both be enabled. Disabling Pseudo Aggregation.');
                pseudoAggSwitch.checked = false;
                pseudoAggSwitch.dispatchEvent(new Event('change'));
            }
        });

        // Prevent enabling mid-round test if pseudo agg is enabled
        if (pseudoAggSwitch) {
            pseudoAggSwitch.addEventListener('change', function() {
                if (pseudoAggSwitch.checked && midRoundTestSwitch.checked) {
                    alert('Pseudo Aggregation and Mid-Round Testing cannot both be enabled. Disabling Mid-Round Testing.');
                    midRoundTestSwitch.checked = false;
                    midRoundTestSwitch.dispatchEvent(new Event('change'));
                }
            });
        }
    }

    function getSettings() {
        return {
            enabled: Settings.enabled
        };
    }

    function isEnabled() {
        return Settings.enabled;
    }

    return {
        initialize,
        getSettings,
        isEnabled
    };
})();

export default MidRoundTestManager;
