from typing import Any
import shutil

class Station:
    """Represents the status information of an associated station."""

    def __init__(self):
        self.info_dict: dict[str, Any] = {}
        self.raw: str | None = None

        # Primary identifiers
        self.mac: str | None = None
        self.flags: list[str] = []

        # Basic STA info
        self.aid: int | None = None
        self.capability: list[str] | None = None
        self.listen_interval: int | None = None
        self.supported_rates: list[float] | None = None
        self.timeout_next: str | None = None

        # Traffic counters
        self.rx_packets: int | None = None
        self.tx_packets: int | None = None
        self.rx_bytes: int | None = None
        self.tx_bytes: int | None = None
        self.rx_airtime: int | None = None
        self.tx_airtime: int | None = None
        self.beacons_count: int | None = None
        self.rx_drop_misc: int | None = None
        self.backlog_packets: int | None = None
        self.backlog_bytes: int | None = None
        self.fcs_error_count: int | None = None
        self.beacon_loss_count: int | None = None
        self.expected_throughput: int | None = None
        self.tx_retry_count: int | None = None
        self.tx_retry_failed: int | None = None

        # Rate / airtime info
        self.tx_bitrate: int | None = None
        self.rx_bitrate: int | None = None
        self.tx_duration: int | None = None
        self.rx_duration: int | None = None

        # PHY layer fields
        self.rx_mcs: int | None = None
        self.tx_mcs: int | None = None
        self.rx_vhtmcs: int | None = None
        self.tx_vhtmcs: int | None = None
        self.rx_he_nss: int | None = None
        self.tx_he_nss: int | None = None
        self.rx_vht_nss: int | None = None
        self.tx_vht_nss: int | None = None
        self.rx_dcm: int | None = None
        self.tx_dcm: int | None = None
        self.rx_guard_interval: int | None = None
        self.tx_guard_interval: int | None = None

        # Signal / timing
        self.signal: int | None = None
        self.avg_signal: int | None = None
        self.avg_beacon_signal: int | None = None
        self.avg_ack_signal: int | None = None
        self.inactive_msec: int | None = None
        self.connected_sec: int | None = None

        # Extra info
        self.rx_rate_info: int | None = None
        self.tx_rate_info: int | None = None
        self.connected_time: int | None = None
        self.mbo_cell_capa: int | None = None
        self.supp_op_classes: str | None = None
        self.min_txpower: int | None = None
        self.max_txpower: int | None = None
        self.ext_capab: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Dump the complete state of the Station object.
        Returns a dictionary of all attributes.
        """
        # Gather all instance variables
        attrs = {
            k: v for k, v in vars(self).items()
        }
        return attrs

    def __str__(self) -> str:
        """Prettier and multi-column tabular string representation of StationBasicInfo."""
        # Try to get the terminal width, or fall back to 100
        try:
            width = shutil.get_terminal_size(fallback=(100, 24)).columns
        except Exception:
            width = 100

        # Collect attributes to display in first table
        skip_attrs = {"info_dict", "raw"}
        attrs = [(attr, getattr(self, attr)) for attr in vars(self) if attr not in skip_attrs]

        # Prepare rows for printing: list of (attr, value) as string (for easy width calc)
        row_tuples = [(str(attr), repr(value)) for attr, value in attrs]

        # Calculate optimal column widths and number of columns
        # Choose a minimum width for each column and fit as many columns as possible
        min_attr_width = max(max(len(attr) for attr, _ in row_tuples), 8)
        min_val_width = max(max(len(val) for _, val in row_tuples), 8)
        col_sep = " ... "
        pair_width = min_attr_width + len(col_sep) + min_val_width

        # Also add left padding for box
        box_pad = 2
        # Try different n_cols from large to small, break if fits
        n_cols = len(row_tuples)
        while n_cols > 1:
            # number of actual _field_ columns (each col holds an attr+val pair)
            # e.g. n_cols = 3 means "attr ... value" * 3
            total_width = n_cols * pair_width + (n_cols - 1) * 2 + 2 * box_pad
            if total_width <= width:
                break
            n_cols -= 1
        else:
            n_cols = 1  # fallback single column

        # Now, make the format string for columns and headings
        attr_fmt = f"{{:<{min_attr_width}}}"
        val_fmt = f"{{:<{min_val_width}}}"
        pair_fmt = attr_fmt + col_sep + val_fmt
        # Compose rows: one row is up to n_cols pairs
        lines = []

        # Decorative boxed title
        title = " Station Basic Info "
        border = "..." * min(width, max(len(title), n_cols * pair_width + (n_cols-1)*2 + 2*box_pad))
        title_line = f"...{border}...\n...{title.center(len(border))}...\n...{border}..."
        lines.append(title_line)

        # Column headings
        head_cells = []
        for _ in range(n_cols):
            head_cells.append(attr_fmt.format("Attribute") + col_sep + val_fmt.format("Value"))
        head_row = ("  " * box_pad) + ("  ".join(head_cells[:n_cols]))
        lines.append(head_row)
        sep_cells = []
        for _ in range(n_cols):
            sep_cells.append("-" * min_attr_width + "-...-" + "-" * min_val_width)
        sep_row = ("  " * box_pad) + ("  ".join(sep_cells[:n_cols]))
        lines.append(sep_row)

        # Table body in columns
        k = 0
        num_rows = (len(row_tuples) + n_cols - 1) // n_cols
        for i in range(num_rows):
            row_cells = []
            for j in range(n_cols):
                idx = i + j * num_rows
                if idx < len(row_tuples):
                    attr, val = row_tuples[idx]
                    cell = pair_fmt.format(attr, val)
                else:
                    cell = " " * pair_width   # blank cell
                row_cells.append(cell)
            row_line = ("  " * box_pad) + ("  ".join(row_cells[:n_cols]))
            lines.append(row_line)

        # Add info_dict as a separate box, if nonempty
        lines.append("\ninfo_dict:")
        if self.info_dict:
            key_width = max(len(str(k)) for k in self.info_dict)
            val2_width = max(len(repr(v)) for v in self.info_dict.values())
            key_width = max(key_width, 4)
            val2_width = max(val2_width, 6)
            dict_col_sep = " ... "
            dict_pair_width = key_width + len(dict_col_sep) + val2_width
            n_dict_cols = max(1, (width - 2 * box_pad) // (dict_pair_width + 2))

            dict_keys = list(self.info_dict.keys())
            dict_vals = list(self.info_dict.values())
            dict_pair_fmt = f"{{:<{key_width}}}{dict_col_sep}{{:<{val2_width}}}"

            # Dict headings
            dict_head_cells = [
                f"{'Key':<{key_width}}{dict_col_sep}{'Value':<{val2_width}}"
                for _ in range(n_dict_cols)
            ]
            dict_head_row = ("  " * box_pad) + ("  ".join(dict_head_cells[:n_dict_cols]))
            lines.append(dict_head_row)
            dict_sep_cells = [
                "-" * key_width + "-...-" + "-" * val2_width
                for _ in range(n_dict_cols)
            ]
            dict_sep_row = ("  " * box_pad) + ("  ".join(dict_sep_cells[:n_dict_cols]))
            lines.append(dict_sep_row)

            dict_num_rows = (len(dict_keys) + n_dict_cols - 1) // n_dict_cols
            for i in range(dict_num_rows):
                row_cells = []
                for j in range(n_dict_cols):
                    idx = i + j * dict_num_rows
                    if idx < len(dict_keys):
                        cell = dict_pair_fmt.format(str(dict_keys[idx]), repr(dict_vals[idx]))
                    else:
                        cell = " " * dict_pair_width
                    row_cells.append(cell)
                row_line = ("  " * box_pad) + ("  ".join(row_cells[:n_dict_cols]))
                lines.append(row_line)
        else:
            lines.append("  (empty)")

        return "\n".join(lines)
