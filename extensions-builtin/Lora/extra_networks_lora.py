import time
import networks
import lora_patches
from modules import extra_networks, shared


class ExtraNetworkLora(extra_networks.ExtraNetwork):

    def __init__(self):
        super().__init__('lora')
        self.active = False
        self.errors = {}
        networks.originals = lora_patches.LoraPatches()

        """mapping of network names to the number of errors the network had during operation"""

    def activate(self, p, params_list):
        t0 = time.time()
        additional = shared.opts.sd_lora
        self.errors.clear()
        if additional != "None" and additional in networks.available_networks and not any(x for x in params_list if x.items[0] == additional):
            p.all_prompts = [x + f"<lora:{additional}:{shared.opts.extra_networks_default_multiplier}>" for x in p.all_prompts]
            params_list.append(extra_networks.ExtraNetworkParams(items=[additional, shared.opts.extra_networks_default_multiplier]))
        if len(params_list) > 0:
            self.active = True
            networks.originals.apply() # apply patches
            if networks.debug:
                shared.log.debug("LoRA activate")
        names = []
        te_multipliers = []
        unet_multipliers = []
        dyn_dims = []
        for params in params_list:
            assert params.items
            names.append(params.positional[0])
            te_multiplier = float(params.positional[1]) if len(params.positional) > 1 else 1.0
            te_multiplier = float(params.named.get("te", te_multiplier))
            unet_multiplier = float(params.positional[2]) if len(params.positional) > 2 else te_multiplier
            unet_multiplier = float(params.named.get("unet", unet_multiplier))
            dyn_dim = int(params.positional[3]) if len(params.positional) > 3 else None
            dyn_dim = int(params.named["dyn"]) if "dyn" in params.named else dyn_dim
            te_multipliers.append(te_multiplier)
            unet_multipliers.append(unet_multiplier)
            dyn_dims.append(dyn_dim)
        t1 = time.time()
        networks.load_networks(names, te_multipliers, unet_multipliers, dyn_dims)
        t2 = time.time()
        if shared.opts.lora_add_hashes_to_infotext:
            network_hashes = []
            for item in networks.loaded_networks:
                shorthash = item.network_on_disk.shorthash
                if not shorthash:
                    continue
                alias = item.mentioned_name
                if not alias:
                    continue
                alias = alias.replace(":", "").replace(",", "")
                network_hashes.append(f"{alias}: {shorthash}")
            if network_hashes:
                p.extra_generation_params["Lora hashes"] = ", ".join(network_hashes)
        if len(names) > 0:
            shared.log.info(f'Applying LoRA: {names} patch={t1-t0:.2f}s load={t2-t1:.2f}s')
        elif self.active:
            self.active = False

    def deactivate(self, p):
        if not self.active and getattr(networks, "originals", None ) is not None:
            networks.originals.undo() # remove patches
            if networks.debug:
                shared.log.debug("LoRA deactivate")
        if self.active and networks.debug:
            shared.log.debug(f"LoRA end: load={networks.timer['load']:.2f}s apply={networks.timer['apply']:.2f}s restore={networks.timer['restore']:.2f}s")
        if self.errors:
            p.comment("Networks with errors: " + ", ".join(f"{k} ({v})" for k, v in self.errors.items()))
            for k, v in self.errors.items():
                shared.log.error(f'LoRA errors: file="{k}" errors={v}')
            self.errors.clear()
