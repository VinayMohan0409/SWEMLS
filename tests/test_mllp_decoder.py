from aki_service.mllp import MLLPDecoder, frame_mllp


def test_mllp_decoder_splits_multiple_frames():
    d = MLLPDecoder()
    msg1 = b"MSH|^~\\&|A\rPID|1||123\r"
    msg2 = b"MSH|^~\\&|B\rPID|1||456\r"
    buf = frame_mllp(msg1) + frame_mllp(msg2)

    frames = d.feed(buf)
    assert [f.hl7 for f in frames] == [msg1, msg2]


def test_mllp_decoder_handles_partial_reads():
    d = MLLPDecoder()
    msg = b"MSH|^~\\&|A\rPID|1||123\r"
    framed = frame_mllp(msg)
    part1 = framed[:5]
    part2 = framed[5:]

    assert d.feed(part1) == []
    frames = d.feed(part2)
    assert len(frames) == 1
    assert frames[0].hl7 == msg
