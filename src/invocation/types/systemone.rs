use std::marker::PhantomData;

use serde_json::Value;

use crate::{
    services::systemone::{HasQuestions, NoQuestions, QuestionSet, SystemOneClient},
    Invocation, InvocationError, NoulCriteria, Question, SystemOneQuestion, SystemOneRequest,
    SystemOneResponse,
};

/// State and named questions for a single System One evaluation.
#[derive(Debug, Clone)]
pub struct SystemOneCall<Q = NoQuestions> {
    state: Value,
    questions: QuestionSet,
    marker: PhantomData<Q>,
}

pub type SystemOneInvocation<Q = NoQuestions> = Invocation<SystemOneCall<Q>>;

impl Invocation<SystemOneCall<NoQuestions>> {
    pub fn systemone(state: impl Into<Value>) -> Self {
        Self::new(SystemOneCall {
            state: state.into(),
            questions: QuestionSet::default(),
            marker: PhantomData,
        })
    }
}

impl<Q> Invocation<SystemOneCall<Q>> {
    /// Add a named question. This also accepts advanced structured questions.
    pub fn question(
        self,
        key: impl Into<String>,
        question: SystemOneQuestion,
    ) -> Invocation<SystemOneCall<HasQuestions>> {
        let key = key.into();
        self.map_request(|mut call| {
            call.questions.insert(key, question);
            SystemOneCall {
                state: call.state,
                questions: call.questions,
                marker: PhantomData,
            }
        })
    }

    pub fn noul(
        self,
        key: impl Into<String>,
        instructions: impl Into<Question>,
    ) -> Invocation<SystemOneCall<HasQuestions>> {
        self.question(key, SystemOneQuestion::noul(instructions))
    }

    pub fn noul_with_criteria(
        self,
        key: impl Into<String>,
        instructions: impl Into<Question>,
        criteria: NoulCriteria,
    ) -> Invocation<SystemOneCall<HasQuestions>> {
        self.question(
            key,
            SystemOneQuestion::noul_with_criteria(instructions, criteria),
        )
    }

    pub fn choice<K, V, C>(
        self,
        key: impl Into<String>,
        instructions: impl Into<Question>,
        criteria: C,
    ) -> Invocation<SystemOneCall<HasQuestions>>
    where
        K: Into<String>,
        V: Into<String>,
        C: IntoIterator<Item = (K, V)>,
    {
        self.question(key, SystemOneQuestion::choice(instructions, criteria))
    }

    pub fn score<V, C>(
        self,
        key: impl Into<String>,
        instructions: impl Into<Question>,
        criteria: C,
    ) -> Invocation<SystemOneCall<HasQuestions>>
    where
        V: Into<String>,
        C: IntoIterator<Item = V>,
    {
        self.question(key, SystemOneQuestion::score(instructions, criteria))
    }
}

impl Invocation<SystemOneCall<HasQuestions>> {
    pub async fn invoke(self) -> Result<SystemOneResponse, InvocationError> {
        self.request
            .questions
            .validate()
            .map_err(InvocationError::InvalidSystemOneQuestion)?;
        let client = SystemOneClient::new(self.client_config)?;
        Ok(client
            .evaluate(SystemOneRequest {
                state: self.request.state,
                model: self.model,
                questions: self.request.questions.questions,
            })
            .await?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SystemOneAnswer;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    #[test]
    fn multiple_questions_keep_their_keys_and_model_is_optional() {
        let invocation = Invocation::systemone("wrong size")
            .noul("refund", "Does the customer ask for a refund?")
            .noul(
                "exchange",
                Question::from("Does the customer ask for an exchange?"),
            );
        assert_eq!(invocation.request.questions.questions.len(), 2);
        assert!(invocation.model.is_none());
        let request = SystemOneRequest {
            state: invocation.request.state,
            model: invocation.model,
            questions: invocation.request.questions.questions,
        };
        let body = serde_json::to_value(request).unwrap();
        assert!(body.get("model").is_none());
        assert_eq!(body["questions"]["refund"]["type"], "noul");
        assert_eq!(body["questions"]["exchange"]["type"], "noul");
    }

    #[tokio::test]
    async fn duplicate_keys_and_invalid_criteria_fail_before_transport() {
        let duplicate = Invocation::systemone("state")
            .noul("same", "First?")
            .noul("same", "Second?")
            .invoke()
            .await;
        assert!(matches!(
            duplicate,
            Err(InvocationError::InvalidSystemOneQuestion(message)) if message.contains("duplicate")
        ));

        let invalid = Invocation::systemone("state")
            .choice("category", "Which?", [("only", "One option")])
            .invoke()
            .await;
        assert!(matches!(
            invalid,
            Err(InvocationError::InvalidSystemOneQuestion(message)) if message.contains("2 to 255")
        ));
    }

    #[tokio::test]
    async fn invoke_posts_native_request_and_decodes_typed_answers() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut connection, _) = listener.accept().unwrap();
            connection
                .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                .unwrap();
            let mut bytes = Vec::new();
            let mut chunk = [0_u8; 4096];
            let (header_end, content_length) = loop {
                let read = connection.read(&mut chunk).unwrap();
                assert!(read > 0);
                bytes.extend_from_slice(&chunk[..read]);
                if let Some(end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                    let header = String::from_utf8_lossy(&bytes[..end]);
                    let length = header
                        .lines()
                        .find_map(|line| {
                            line.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .and_then(|value| value.trim().parse::<usize>().ok())
                        })
                        .unwrap();
                    break (end + 4, length);
                }
            };
            while bytes.len() - header_end < content_length {
                let read = connection.read(&mut chunk).unwrap();
                assert!(read > 0);
                bytes.extend_from_slice(&chunk[..read]);
            }
            let request: Value =
                serde_json::from_slice(&bytes[header_end..header_end + content_length]).unwrap();
            let body = r#"{"model":"diy-jev-0.1.0","answers":{"refund":{"type":"noul","noul":0.92},"department":{"type":"choice","choice":"returns","confidence":0.85,"probabilities":{"returns":0.85,"shipping":0.15}}},"usage":{"input_tokens":128,"output_tokens":3}}"#;
            write!(
                connection,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            )
            .unwrap();
            request
        });

        let response = Invocation::systemone("wrong size")
            .base_url(format!("http://{address}"))
            .noul("refund", "Is a refund requested?")
            .choice(
                "department",
                "Which department?",
                [("returns", "Returns"), ("shipping", "Shipping")],
            )
            .invoke()
            .await
            .unwrap();
        let request = server.join().unwrap();
        assert_eq!(request["state"], "wrong size");
        assert!(request.get("model").is_none());
        assert_eq!(request["questions"]["department"]["type"], "choice");
        assert!(matches!(
            response.answers.get("refund"),
            Some(SystemOneAnswer::Noul(answer)) if answer.noul == 0.92
        ));
        assert_eq!(response.usage.input_tokens, 128);
    }
}

impl Invocation<SystemOneCall<NoQuestions>> {
    pub(crate) fn with_questions(
        self,
        questions: QuestionSet,
    ) -> Invocation<SystemOneCall<HasQuestions>> {
        self.map_request(|call| SystemOneCall {
            state: call.state,
            questions,
            marker: PhantomData,
        })
    }
}
