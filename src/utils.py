def enforce_aspect_ratio(cx: float, cy: float, w: float, h: float, target_ratio: float) -> tuple[float, float, float, float]:
        """
        Adjusts width or height so that the box matches the target aspect ratio,
        centered around (cx, cy), by expanding the shorter side.

        Args:
            cx: Center x
            cy: Center y
            w: Current width
            h: Current height
            target_ratio: Desired aspect ratio (width / height)

        Returns:
            (cx, cy, adjusted_w, adjusted_h)
        """
        current_ratio = w / h if h != 0 else 0

        if current_ratio > target_ratio:
            # Too wide → increase height
            h = w / target_ratio
        else:
            # Too tall → increase width
            w = h * target_ratio

        return cx, cy, w, h