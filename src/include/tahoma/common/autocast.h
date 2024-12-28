#include <tahoma.h>
#include <ATen/autocast_mode.h>


namespace tahoma {

    /**
     * @brief A guard to enable or disable autocast for a device
     * 
     * This class is used to enable or disable autocast for a device. It is used to temporarily enable or disable autocast
     */
    class AutoCastGuard {
    protected:
        torch::DeviceType device_type;
        bool previous_flag = false;
        bool current_flag = false;
    public:
        AutoCastGuard() = default;
        AutoCastGuard(torch::DeviceType device_type, bool enabled=false):
            device_type(device_type),
            previous_flag(torch::autocast::is_autocast_enabled(device_type)),
            current_flag(enabled)
        {
            if (current_flag != previous_flag){
                torch::autocast::set_autocast_enabled(device_type, current_flag);
            }
        }
        bool is_enabled() const {
            return current_flag;
        }

        ~AutoCastGuard() {
            if (current_flag != previous_flag){
                // restore the previous state
                torch::autocast::set_autocast_enabled(device_type, previous_flag);
            }
        }

        static void clear_cache() {
            torch::autocast::clear_cache();
        }
        static void set_dtype(torch::DeviceType device_type, at::ScalarType dtype) {
            torch::autocast::set_autocast_dtype(device_type, dtype);
        }
        static at::ScalarType get_dtype(torch::DeviceType device_type) {
            return torch::autocast::get_autocast_dtype(device_type);
        }
    };
}