use super::{output_item, OutputItem};
use code_mode_protocol::FunctionCallOutputContentItem;

#[test]
fn audio_output_round_trips_without_text_projection() {
    let item = OutputItem::Audio {
        audio_url: "data:audio/wav;base64,AAAA".to_string(),
    };
    assert_eq!(
        output_item(item),
        FunctionCallOutputContentItem::InputAudio {
            audio_url: "data:audio/wav;base64,AAAA".to_string(),
        }
    );
}
